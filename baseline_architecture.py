import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import numpy as np
import cv2
from torch.utils.data import DataLoader
import os
import torchvision

import wandb


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, dataset_info):
        super(CelebADataset, self).__init__()

        condition_size = dataset_info['condition_size']

        self.dataset_info = dataset_info

        self.X = os.listdir(dataset_info['source'])
        train_len = int(len(self.X) * dataset_info['train_split'])
        
        self.train_X = self.X[0:train_len]
        self.test_X = self.X[train_len:len(self.X)]

        self.Y = os.listdir(dataset_info['target'])
        self.train_Y = self.Y[0:train_len]
        self.test_Y = self.Y[train_len:len(self.Y)]

        if dataset_info['condition'] is not None:
            self.condition_data = np.load(dataset_info['condition'])
        else:
            self.condition_data = np.ones((len(self.X), 15))
        self.condition_data = torch.from_numpy(self.condition_data).float().view(-1, condition_size)

        self.train_condition_data = self.condition_data[0:train_len]
        self.test_condition_data = self.condition_data[train_len:len(self.condition_data)]

        assert len(self.X) == len(self.Y) == self.condition_data.shape[0], 'Number of samples in X, Y and condition data must be equal'

        self.num_samples = self.train_condition_data.shape[0]

        self.transforms = {
            'to_tensor': torchvision.transforms.ToTensor(),
            'normalize_original': torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            'normalize_sketch': torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
            'resize_original': torchvision.transforms.Resize(dataset_info['output_size']),
            'resize_sketch': torchvision.transforms.Resize(dataset_info['input_size']),
        }


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sketch_path = os.path.join(self.dataset_info['source'], self.train_X[idx])
        sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)

        original_path = os.path.join(self.dataset_info['target'], self.train_Y[idx])
        original = cv2.imread(original_path)

        sketch = self.transforms['to_tensor'](sketch)
        original = self.transforms['to_tensor'](original)

        original = self.transforms['normalize_original'](original)
        sketch = self.transforms['normalize_sketch'](sketch)

        original = self.transforms['resize_original'](original)
        sketch = self.transforms['resize_sketch'](sketch)

        return sketch, original, self.train_condition_data[idx]
    
    def get_test_data(self):
        X = []
        Y = []
        for sketch_path, original_path in zip(self.test_X, self.test_Y):
            sketch_path = os.path.join(self.dataset_info['source'], sketch_path)
            sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)

            original_path = os.path.join(self.dataset_info['target'], original_path)
            original = cv2.imread(original_path)

            sketch = self.transforms['to_tensor'](sketch)
            original = self.transforms['to_tensor'](original)

            original = self.transforms['normalize_original'](original)
            sketch = self.transforms['normalize_sketch'](sketch)

            original = self.transforms['resize_original'](original)
            sketch = self.transforms['resize_sketch'](sketch)

            X.append(sketch)
            Y.append(original)
        
        return torch.stack(X), torch.stack(Y), self.test_condition_data
        

class AutoEncoder(nn.Module):
    def __init__(self, input_size, output_size, condition_size=1024, num_img=None, dataset_info={
        'source': 'sketches',
        'target': 'original',
        'condition': 'conditions.npy',
        'train_split': 0.95,
    }, model_config={
        'condition': True,
        'condition_type': 'mul',
        'skip_connection': True,
    }):
        super(AutoEncoder, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
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
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 64, 3)
        self.max_pool = nn.MaxPool2d(2, 2)

        self.condition_projection = nn.Linear(15, condition_size)

        # bottleneck and projection layers
        self.bottle_neck = nn.LazyLinear(condition_size)
        self.projection_layer = nn.Linear(condition_size, self.projection_size)
        
        # Decoder layers

        self.resize_conv_block_1 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=0))
        
        self.resize_conv_block_2 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0))
        
        self.resize_conv_block_3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0)
        )

        self.resize_conv_block_4 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=0)
        )

        # # dummy pass to initialize watch:
        # with torch.no_grad():
        #     dummy_batch = self.train_X[:1]
        #     dummy_cond = self.train_condition[:1]
        #     _ = self.forward(dummy_batch, dummy_cond)

        # wandb.watch(self, self.loss, log="all")

        self.PSNR_train = PeakSignalNoiseRatio()
        self.PSNR_test = PeakSignalNoiseRatio()

        self.SSIM_train = StructuralSimilarityIndexMeasure()
        self.SSIM_test = StructuralSimilarityIndexMeasure()

        self.to(self.device)

    def encoder(self, x, condition):
        # Encoder Forward Pass
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.max_pool(x)
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
        x = self.resize_conv_block_1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.resize_conv_block_2(x)
        x = F.relu(x)
        x = self.resize_conv_block_3(x)
        x = F.relu(x)
        x = self.resize_conv_block_4(x)
        x = F.relu(x)
        return x
    
    def forward(self, input, condition):
        x = self.encoder(input, condition)
        x = self.decoder(x, input)
        return x
    
    def loss(self, ground_truth, output):
        loss = F.mse_loss(ground_truth, output)
        return loss

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

                self.PSNR_train.update(output_train, targets)
                self.SSIM_train.uptdate(output_train, targets)
                

                loss_train.backward()
                optimizer.step()

                print(f'Epoch {epoch} Batch {batch_idx} Loss: {loss_train}')
            psnr_train = self.PSNR_train.compute()
            ssim_train = self.SSIM_train.compute()

            self.PSNR_train.reset()
            self.SSIM_train.reset()
                # use this block to calculate all test set metrics to avoid affecting model
            if gen_images:
                with torch.no_grad():
                    # generate test images
                    images = os.listdir(gen_images)
                    images = [cv2.resize(cv2.imread(os.path.join(gen_images, image), cv2.IMREAD_GRAYSCALE), self.input_size) for image in images]
                    images = np.array(images)
                    images = images - np.mean(images) / np.std(images)
                    images = torch.from_numpy(images).float().view(-1, 1, self.input_size, self.input_size).to(self.device)
                    
                    if gen_condition:
                        conditions = np.load(gen_condition)
                        conditions = torch.from_numpy(conditions).float().view(-1, 15).to(self.device)
                    else:
                        conditions = torch.ones((1, 15)).to(self.device)
                    gen_images = self.forward(images, conditions)
                    gen_images = gen_images.numpy().reshape(-1, 252, 252, 3)
                    for i, image in enumerate(gen_images):
                        if not os.path.exists('gen_images'):
                            os.mkdir('gen_images')
                        cv2.imwrite(f'gen_images/{i}.jpg', image)
                
            print(f'Epoch {epoch} Loss: {loss_train}')

            with torch.no_grad():
                test_X = test_X.to(self.device)
                test_Y = test_Y.to(self.device)
                test_condition = test_condition.to(self.device)
                output_valid = self.forward(test_X, test_condition)
                loss_valid = self.loss(test_Y, output_valid)
                print(f'Validation Loss: {loss_valid}')
                self.PSNR_test.updata(output_valid, test_Y)
                self.SSIM_test.update(output_valid, test_Y)
                psnr_valid = self.PSNR_test.compute()
                ssim_valid = self.SSIM_test.compute()
                self.PSNR_test.reset()
                self.SSIM_test.reset()
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
    ae = AutoEncoder(input_size=(128, 128), output_size=(252, 252), condition_size=1024)
    ae.train(epochs=10, batch_size=128, save_path='model.pth')
