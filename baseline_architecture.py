import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
import numpy as np
import cv2
from torch.utils.data import DataLoader
import os
import torchvision

import wandb

class unetBlocks:
    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, dataset_info):
        super(CelebADataset, self).__init__()

        num_img = dataset_info['num_img']

        condition_size = dataset_info['condition_size']


        self.dataset_info = dataset_info

        self.X = os.listdir(dataset_info['source'])[:num_img]
        train_len = int(len(self.X) * dataset_info['train_split'])
        
        self.train_X = self.X[0:train_len]
        self.test_X = self.X[train_len:len(self.X)]

        self.Y = os.listdir(dataset_info['target'])[:num_img]
        self.train_Y = self.Y[0:train_len]
        self.test_Y = self.Y[train_len:len(self.Y)]

        if dataset_info['condition'] is not None:
            self.condition_data = np.load(dataset_info['condition'])
        else:
            self.condition_data = np.ones((len(self.X), 15))
        self.condition_data = torch.from_numpy(self.condition_data).float().view(-1, condition_size)[:num_img]

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
    
    def get_validation_sample(self, sample_size):
        X = []
        Y = []

        print('LENGTH OF TEXT: ', len(self.test_X))

        sample_indices = np.random.choice(len(self.test_X),sample_size,False)
        sample_X = np.array(self.test_X)[sample_indices]
        sample_Y = np.array(self.test_Y)[sample_indices]
        sample_condition_vec = np.array(self.test_condition_data)[sample_indices]

        for sketch_path, original_path in zip(sample_X, sample_Y):
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
        
        return torch.stack(X), torch.stack(Y), sample_condition_vec

        
        # get random permutation of images


        

class AutoEncoder(nn.Module):
    def __init__(self, input_size, output_size, num_img=None, dataset_info={
        'source': 'sketches',
        'target': 'original',
        'condition': 'conditions.npy',
        'num_img': 100,
        'train_split': 0.95,
    }, model_config={
        'condition': True,
        'condition_type': 'mul',
        'skip_connection': True,
    }):
        super(AutoEncoder, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.art = wandb.Artifact("baseline", type="model")
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
        self.projection_size = 1024
        self.model_config = model_config

        self.dataset = CelebADataset({
            'source': dataset_info['source'],
            'target': dataset_info['target'],
            'condition': dataset_info['condition'],
            'train_split': dataset_info['train_split'],
            'input_size': self.input_size,
            'output_size': output_size,
            'num_img': dataset_info['num_img'],
            'condition_size': 15,
        })


        # Encoder layers
        self.encoder_conv1 = unetBlocks.conv_block(1, 64)
        self.encoder_conv2 = unetBlocks.conv_block(64, 128)
        self.encoder_conv3 = unetBlocks.conv_block(128, 256)
        self.encoder_conv4 = unetBlocks.conv_block(256, 512)

        self.max_pool = nn.MaxPool2d(2)

        self.condition_projection = nn.Linear(15, self.projection_size)
        
        # Decoder layers
        self.decoder_conv1 = unetBlocks.conv_block(512, 256)
        self.decoder_conv2 = unetBlocks.conv_block(256, 128)
        self.decoder_conv3 = unetBlocks.conv_block(128, 64)

        self.decoder_upconv1 = unetBlocks.upconv_block(512, 256)
        self.decoder_upconv2 = unetBlocks.upconv_block(256, 128)
        self.decoder_upconv3 = unetBlocks.upconv_block(128, 64)

        # Output
        self.output_conv = nn.Conv2d(64, 3, kernel_size=1)

        self.to(self.device)
    
    def forward(self, input, condition):
        enc1 = self.encoder_conv1(input)
        enc2 = self.encoder_conv2(self.max_pool(enc1))
        enc3 = self.encoder_conv3(self.max_pool(enc2))
        enc4 = self.encoder_conv4(self.max_pool(enc3))

        condition = self.condition_projection(condition)
        condition = F.relu(condition)
        condition = condition.view(-1, 1, 32, 32)

        conditioned_enc4 = enc4 * condition

        dec1 = self.decoder_upconv1(conditioned_enc4)
        dec1 = torch.cat((enc3, dec1), dim=1)
        dec1 = self.decoder_conv1(dec1)

        dec2 = self.decoder_upconv2(dec1)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder_conv2(dec2)

        dec3 = self.decoder_upconv3(dec2)
        dec3 = torch.cat((enc1, dec3), dim=1)
        dec3 = self.decoder_conv3(dec3)

        out = self.output_conv(dec3)
        return out
    
    def loss(self, ground_truth, output):
        loss = F.mse_loss(ground_truth, output)
        return loss

    def PSNR(self, output, ground_truth):
        psnr = peak_signal_noise_ratio(output, ground_truth)
        return psnr
    
    def SSIM(self, output, ground_truth):
        ssim = structural_similarity_index_measure(output, ground_truth)
        return ssim

    def train(self, epochs, batch_size, save_path, gen_images = None, gen_condition = None, gen_images_input = None, gen_images_output = None):

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        
        for epoch in range(epochs):
            epoch_loss_train = []
            epoch_psnr_train = []
            epoch_ssim_train = []
            for batch_idx, (images, conditions, targets) in enumerate(self.dataloader):
                print('\n\n\nBATCH INDEX: ', batch_idx)
                images, targets, conditions = images.to(self.device), conditions.to(self.device), targets.to(self.device)
                optimizer.zero_grad()

                output_train = self.forward(images, conditions)
                
                wandb.watch(self, log="all")

                loss_train = self.loss(targets, output_train)

                psnr_train = self.PSNR(output_train, targets)
                ssim_train = self.SSIM(output_train, targets)

                epoch_loss_train.append(loss_train)
                epoch_psnr_train.append(psnr_train)
                epoch_ssim_train.append(ssim_train)
                

                loss_train.backward()
                optimizer.step()

                if batch_idx%10==0:
                    print(f'Epoch {epoch+1}/{epochs} Batch {batch_idx+1} Loss: {loss_train}')

                # use this block to calculate all test set metrics to avoid affecting model

            # generate images from two sets of sketches: holdout and cuhk
            if gen_images is not None: # boolean
                with torch.no_grad():
                    # generate test images
                    image_files = os.listdir(gen_images_input)
                    images = [cv2.resize(cv2.imread(os.path.join(gen_images_input, image), cv2.IMREAD_GRAYSCALE), self.input_size) for image in image_files]
                    images = np.array(images)
                    images = torch.from_numpy(images).float().view(-1, 1, self.input_size[0], self.input_size[1])
                    images = self.dataset.transforms['normalize_sketch'](images)
                    images = images.to(self.device)
                    
                    if gen_condition:
                        conditions = np.load(gen_condition)
                        conditions = torch.from_numpy(conditions).float().view(-1, 15).to(self.device)
                    else:
                        conditions = torch.randint(0, 2, (images.shape[0], 15)).float().to(self.device)
                    output_images = self.forward(images, conditions)
                    output_images = output_images.cpu().numpy().reshape(-1, self.output_size[0], self.output_size[1], 3)

                    output_dir = gen_images_output + f'/epoch_{epoch}'
                    if not os.path.exists(output_dir):
                            os.mkdir(output_dir)

                    for i, image in enumerate(output_images):
                        
                        cv2.imwrite(output_dir + f'/{i}.jpg', image)
                    
                print(f'Epoch {epoch} Loss: {loss_train}')
                
            test_X, test_Y, test_condition = self.dataset.get_validation_sample(sample_size=32)
            
            # convert test_condition back to tensor
            test_condition= torch.from_numpy(test_condition).float().view(-1, 15)

            with torch.no_grad():
                test_X = test_X.to(self.device)
                test_Y = test_Y.to(self.device)
                test_condition = test_condition.to(self.device)
                output_valid = self.forward(test_X, test_condition)
                loss_valid = self.loss(test_Y, output_valid)
                print(f'Validation Loss: {loss_valid}')

                psnr_valid = self.PSNR(output_valid, test_Y)
                ssim_valid = self.SSIM(output_valid, test_Y)
            self.save_model(save_path)

            
            mean_epoch_loss_train = sum(epoch_loss_train)/len(epoch_loss_train)
            mean_epoch_psnr_train = sum(epoch_psnr_train)/len(epoch_psnr_train)
            mean_epoch_ssim_train = sum(epoch_ssim_train)/len(epoch_ssim_train)

            wandb.log({
                "epoch": epoch+1,
                "train-reconstruction-loss": mean_epoch_loss_train,
                'valid-reconstruction-loss': loss_valid,
                "train-PSNR": mean_epoch_psnr_train,
                'valid-PSNR': psnr_valid,
                'train-SSIM': mean_epoch_ssim_train,
                'valid-SSIM': ssim_valid
             })
            
        wandb.log_artifact(self.art)
        wandb.finish()

    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))

if __name__ == '__main__':
    dataset_info={
    'source': 'model_input/sketches',
    'target': 'model_input/original',
    'condition': 'model_input/conditions.npy',
    'num_img': 100,
    'train_split': 0.8,
    }

    input_size = [256, 256]
    output_size = [256, 256]

    AE = AutoEncoder(input_size=input_size,
        output_size=output_size,
        dataset_info=dataset_info
    )

    AE.train(
        epochs=1, 
        batch_size=32, 
        save_path='model.pth',
        gen_images=True,
        gen_images_input='generation_input/sketches_holdout',
        gen_images_output='generation_output')

