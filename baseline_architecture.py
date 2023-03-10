import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
import numpy as np
import cv2
from torch.utils.data import DataLoader
import os

import wandb

class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, dataset_info):
        super(CelebADataset, self).__init__()

        input_size = dataset_info['input_size']
        output_size = dataset_info['output_size']
        condition_size = dataset_info['condition_size']

        self.X = np.load(dataset_info['source'])[:1000]
        self.X = self.X - np.mean(self.X) / np.std(self.X)
        self.X = torch.from_numpy(self.X).float().view(-1, 1, input_size[0], input_size[1])
        train_len = int(len(self.X) * dataset_info['train_split'])
        
        self.train_X = self.X[0:train_len]
        self.test_X = self.X[train_len:len(self.X)]

        self.Y = np.load(dataset_info['target'])[:1000]
        self.Y = self.Y - np.mean(self.Y) / np.std(self.Y)
        self.Y = torch.from_numpy(self.Y).float().view(-1, 3, output_size[0], output_size[1])
        self.train_Y = self.Y[0:train_len]
        self.test_Y = self.Y[train_len:len(self.Y)]

        if dataset_info['condition'] is not None:
            self.condition_data = np.load(dataset_info['condition'])
        else:
            self.condition_data = np.ones((len(self.X), 15))[:1000]
        self.condition_data = torch.from_numpy(self.condition_data).float().view(-1, condition_size)

        self.train_condition_data = self.condition_data[0:train_len]
        self.test_condition_data = self.condition_data[train_len:len(self.condition_data)]

        assert self.X.shape[0] == self.Y.shape[0] == self.condition_data.shape[0], 'Number of samples in X, Y and condition data must be equal'

        self.num_samples = self.train_condition_data.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.train_X[idx], self.train_Y[idx], self.train_condition_data[idx]
    
    def get_test_data(self):
        return self.test_X, self.test_Y, self.test_condition_data

class AutoEncoder(nn.Module):
    def __init__(self, input_size, output_size, condition_size=1024, num_img=None, dataset_info={
        'source': 'sketches.npy',
        'target': 'originals.npy',
        'condition': None,
        'train_split': 0.8,
    }, model_config={
        'condition': True,
        'condition_type': 'mul',
        'skip_connection': True,
    }):
        super(AutoEncoder, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            # self.device = torch.device('mps')
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        self.art = wandb.Artifact("baseline", type="model")
        # art.add_file("test_model_model_weights.pt")
        #  WANDB INITIALIZATION
        wandb.init(
            # set the wandb project where this run will be logged
            project="sketch-VAE",
            
            # track hyperparameters and run metadata
            config={
            # "learning_rate": 0.02,
            "architecture": "VAE",
            "dataset": "beta",
            
            }
        )

        self.input_size = input_size
        self.output_size = output_size
        self.projection_size = input_size[0] * input_size[1]
        self.condition_size = condition_size
        self.model_config = model_config

        self.dataset = CelebADataset({
            'source': dataset_info['source'],
            'target': dataset_info['target'],
            'condition': dataset_info['condition'],
            'train_split': dataset_info['train_split'],
            'input_size': self.input_size,
            'output_size': output_size,
            'condition_size': 15,
        })


        # Encoder layers
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 4, 3)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(4, 3, 3)
        self.conv5 = nn.Conv2d(3, 3, 3)

        self.condition_projection = nn.Linear(15, condition_size)

        # bottleneck and projection layers
        self.bottle_neck = nn.LazyLinear(condition_size)
        self.projection_layer = nn.Linear(condition_size, self.projection_size)
        
        # Decoder layers
        self.deconv1 = nn.ConvTranspose2d(2, 8, 3)
        self.deconv2 = nn.ConvTranspose2d(8, 16, 3)
        self.deconv3 = nn.ConvTranspose2d(16, 3, 3)
        self.up_sample = nn.ConvTranspose2d(3, 3, 2, stride=2)




        # # dummy pass to initialize watch:
        # with torch.no_grad():
        #     dummy_batch = self.train_X[:1]
        #     dummy_cond = self.train_condition[:1]
        #     _ = self.forward(dummy_batch, dummy_cond)

        # wandb.watch(self, self.loss, log="all")

        self.to(self.device)

    def encoder(self, x, condition):
        # Encoder Forward Pass
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        # high dimensional projection through linear layer
        x = torch.flatten(x, 1)
        x = self.bottle_neck(x)
        x = F.relu(x)
        condition = self.condition_projection(condition)
        if self.model_config['condition_type'] == 'add':
            x = torch.add(x, condition)
        else:
            x = torch.mul(x, condition)
        x = self.projection_layer(x)
        x = F.relu(x)
        return x
    
    def decoder(self, encoded_representation, input):
        x = encoded_representation.view(-1, 1, self.input_size[0], self.input_size[1])
        if self.model_config['skip_connection']:
            x = torch.cat((x, input), dim=1)
        else:
            x = x
        x = self.deconv1(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = F.relu(x)
        x = self.deconv3(x)
        x = F.relu(x)
        x = self.up_sample(x)
        x = F.relu(x)
        return x
    
    def forward(self, input, condition):
        x = self.encoder(input, condition)
        x = self.decoder(x, input)
        return x
    
    def loss(self, ground_truth, output):
        loss = F.mse_loss(ground_truth, output)
        return loss

    def PSNR(self, output, ground_truth):
        psnr = peak_signal_noise_ratio(output, ground_truth)
        return psnr
    
    def SSIM(self, output, ground_truth):
        ssim = structural_similarity_index_measure(output, ground_truth)
        return ssim

    def train(self, epochs, batch_size, save_path, gen_images = None, gen_condition = None):

        # # dummy pass to initialize watch:
        # with torch.no_grad():
        #     dummy_batch = self.train_X[:1]
        #     # dummy_cond = np.zeros((len(self.X), self.condition_size))
        #     dummy_cond = self.train_condition[:1]
        #     _ = self.forward(dummy_batch, dummy_cond)

        # wandb.watch(self, self.loss, log="all")

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        test_X, test_Y, test_condition = self.dataset.get_test_data()
        for epoch in range(epochs):
            for batch_idx, (images, conditions, targets) in enumerate(self.dataloader):
                images, targets, conditions = images.to(self.device), conditions.to(self.device), targets.to(self.device)
                optimizer.zero_grad()

                output_train = self.forward(images, conditions)
                
                wandb.watch(self, log="all")

                loss_train = self.loss(targets, output_train)

                psnr_train = self.PSNR(output_train, targets)
                ssim_train = self.SSIM(output_train, targets)
                

                loss_train.backward()
                optimizer.step()

                print(f'Epoch {epoch} Batch {batch_idx} Loss: {loss_train}')

                # use this block to calculate all test set metrics to avoid affecting model
            if gen_images:
                with torch.no_grad():
                    # generate test images
                    images = os.listdir(gen_images)
                    images = [cv2.resize(cv2.imread(os.path.join(gen_images, image), cv2.IMREAD_GRAYSCALE), (self.input_size, self.input_size)) for image in images]
                    images = np.array(images)
                    images = images - np.mean(images) / np.std(images)
                    images = torch.from_numpy(images).float().view(-1, 1, self.input_size, self.input_size).to(self.device)
                    
                    if gen_condition:
                        conditions = np.load(gen_condition)
                        conditions = torch.from_numpy(conditions).float().view(-1, 15).to(self.device)
                    else:
                        conditions = torch.ones((1, 15)).to(self.device)
                    gen_images = self.forward(images, conditions)
                    gen_images = gen_images.numpy().reshape(-1, 268, 268, 3)
                    for i, image in enumerate(gen_images):
                        if not os.path.exists('gen_images'):
                            os.mkdir('gen_images')
                        cv2.imwrite(f'gen_images/{i}.jpg', image)
                
            print(f'Epoch {epoch} Loss: {loss_train}')

            with torch.no_grad():
                output_valid = self.forward(test_X, test_condition)
                loss_valid = self.loss(test_Y, output_valid)
                print(f'Validation Loss: {loss_valid}')

                psnr_valid = self.PSNR(output_valid, test_Y)
                ssim_valid = self.PSNR(output_valid, test_Y)
            self.save_model(save_path)
       
            wandb.log({
                "epoch": epoch+1,
                "train-reconstruction-loss": loss_train,
                'valid-reconstruction-loss': loss_valid,
                "train-PSNR": psnr_train,
                'valid-PSNR': psnr_valid,
                'train-SSIM': ssim_train,
                'valid-SSIM': ssim_valid
             })
            
        wandb.log_artifact(self.art)
        wandb.finish()

    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))

if __name__ == '__main__':
    ae = AutoEncoder(input_size=(128, 128), output_size=(268, 268), condition_size=1024)
    ae.train(epochs=10, batch_size=350, save_path='model.pth')
