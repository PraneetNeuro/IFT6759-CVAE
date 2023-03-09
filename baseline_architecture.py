import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
import numpy as np
import cv2

import wandb


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
        
        self.to(self.device)


        self.art = wandb.Artifact("test-model", type="model")
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
        self.condition_size = condition_size # size of the condition vector and the bottleneck layer
        self.model_config = model_config

        self.X = np.load(dataset_info['source'])[:num_img]
        self.X = self.X - np.mean(self.X) / np.std(self.X)
        self.X = torch.from_numpy(self.X).float().view(-1, 1, self.input_size[0], self.input_size[1]).to(self.device)

        self.Y = np.load(dataset_info['target'])[:num_img]
        self.Y = self.Y - np.mean(self.Y) / np.std(self.Y)
        # self.Y = np.array([cv2.resize(img, self.output_size) for img in self.Y])
        self.Y = torch.from_numpy(self.Y).float().view(-1, 3, self.output_size[0], self.output_size[1]).to(self.device)

        if dataset_info['condition'] is not None:
            # self.condition_data = np.load(dataset_info['condition'])
            self.condition_data = np.ones((4900, self.condition_size))
        else:
            if self.model_config['condition_type'] == 'add':
                self.condition_data = np.zeros((len(self.X), self.condition_size))
            else:
                self.condition_data = np.ones((len(self.X), self.condition_size))
        self.condition_data = torch.from_numpy(self.condition_data).float().view(-1, self.condition_size).to(self.device)

        self.train_X = self.X[:int(len(self.X) * dataset_info['train_split'])]
        self.train_Y = self.Y[:int(len(self.Y) * dataset_info['train_split'])]
        self.train_condition = self.condition_data[:int(len(self.condition_data) * dataset_info['train_split'])]

        self.test_X = self.X[int(len(self.X) * dataset_info['train_split']):]
        self.test_Y = self.Y[int(len(self.Y) * dataset_info['train_split']):]
        self.test_condition = self.condition_data[int(len(self.condition_data) * dataset_info['train_split']):]


        # Encoder layers
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 4, 3)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(4, 3, 3)
        self.conv5 = nn.Conv2d(3, 3, 3)

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
        if not self.model_config['condition']:
            condition = torch.ones(x.size())
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

    def train(self, epochs, batch_size, save_path, gen_images = None):

        # # dummy pass to initialize watch:
        # with torch.no_grad():
        #     dummy_batch = self.train_X[:1]
        #     # dummy_cond = np.zeros((len(self.X), self.condition_size))
        #     dummy_cond = self.train_condition[:1]
        #     _ = self.forward(dummy_batch, dummy_cond)

        # wandb.watch(self, self.loss, log="all")

        

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(epochs):
            for i in range(0, len(self.train_X), batch_size):
                batch_X = self.train_X[i:i+batch_size]
                batch_Y = self.train_Y[i:i+batch_size]
                batch_condition = self.train_condition[i:i+batch_size]
                optimizer.zero_grad()

                output_train = self.forward(batch_X, batch_condition)
                
                wandb.watch(self, log="all")

                loss_train = self.loss(batch_Y, output_train)

                psnr_train = self.PSNR(output_train, batch_Y)
                ssim_train = self.SSIM(output_train, batch_Y)
                

                loss_train.backward()
                optimizer.step()

                # use this block to calculate all test set metrics to avoid affecting model
                if gen_images:
                    with torch.no_grad():

                        # generate test images

                        print(f'Epoch {epoch} Batch {i} Loss: {loss_train}')
                        test_img = cv2.resize(cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE), (128, 128)) 
                        test_img = test_img - np.mean(test_img) / np.std(test_img)
                        test_img = torch.from_numpy(test_img).float().view(-1, 1, self.input_size[0], self.input_size[1]).to(self.device)
                        test_condition = torch.ones((1, self.condition_size)).to(self.device)
                        output_im = self.forward(test_img, test_condition)
                        output_im = output_im.numpy().reshape(268, 268, 3)
                        cv2.imwrite(f'sample_outputs/output_{epoch}_{i}.jpg', output_im)
                

                    


            print(f'Epoch {epoch} Loss: {loss_train}')

            with torch.no_grad():
                output_valid = self.forward(self.test_X, self.test_condition)
                loss_valid = self.loss(self.test_Y, output_valid)
                print(f'Validation Loss: {loss_valid}')

                psnr_valid = self.PSNR(output_valid, self.test_Y)
                ssim_valid = self.PSNR(output_valid, self.test_Y)
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
