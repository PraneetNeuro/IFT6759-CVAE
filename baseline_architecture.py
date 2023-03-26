import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
import numpy as np
import cv2
from torch.utils.data import DataLoader
import os

from celeba import CelebADataset

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



class AutoEncoder(nn.Module):
    def __init__(self, input_size, output_size, num_img=None, dataset_info={
        'source': 'sketches',
        'target': 'original',
        'condition': 'conditions.npy',
        'num_img': 29999,
        'train_split': 0.9999,
    }, model_config={
        'condition': True,
        'condition_type': 'mul',
        'skip_connection': True,
    }):
        super(AutoEncoder, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('mps')

        # first arg denotes model version
        self.model_artifact = wandb.Artifact("baseline_test_versioning", type="model")

        
        #  WANDB INITIALIZATION
        wandb.init(
            # set the wandb project where this run will be logged
            project="sketch-VAE",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": 0.02,
            "architecture": "VAE",
            "dataset": "beta",
            'epochs': 8,
            'batch_size': 32,
            
            }
        )
        dataset_artifact = wandb.Artifact("test_data_version", "dataset")
        dataset_artifact.add_dir("model_input/sketches/")
        dataset = wandb.use_artifact(dataset_artifact)

        self.input_size = input_size
        self.output_size = output_size
        self.projection_size = 256
        self.model_config = model_config

        self.dataset = CelebADataset({
            'source': dataset_info['source'],
            'target': dataset_info['target'],
            'condition': dataset_info['condition'],
            'train_split': dataset_info['train_split'],
            'input_size': self.input_size,
            'output_size': output_size,
            'num_img': dataset_info['num_img'],
            'condition_size': 1,
        })


        # Encoder layers
        self.encoder_conv1 = unetBlocks.conv_block(1, 32)
        self.encoder_conv2 = unetBlocks.conv_block(32, 64)
        self.encoder_conv3 = unetBlocks.conv_block(64, 128)
        self.encoder_conv4 = unetBlocks.conv_block(128, 256)

        self.max_pool = nn.MaxPool2d(2)

        self.condition_projection = nn.Linear(1, self.projection_size)
        
        # Decoder layers
        self.decoder_conv1 = unetBlocks.conv_block(256, 128)
        self.decoder_conv2 = unetBlocks.conv_block(128, 64)
        self.decoder_conv3 = unetBlocks.conv_block(64, 32)

        self.decoder_upconv1 = unetBlocks.upconv_block(257, 128)
        self.decoder_upconv2 = unetBlocks.upconv_block(128, 64)
        self.decoder_upconv3 = unetBlocks.upconv_block(64, 32)

        # Output
        self.output_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        self.to(self.device)
    
    def forward(self, input, condition):
        enc1 = self.encoder_conv1(input)
        enc2 = self.encoder_conv2(self.max_pool(enc1))
        enc3 = self.encoder_conv3(self.max_pool(enc2))
        enc4 = self.encoder_conv4(self.max_pool(enc3))

        condition = self.condition_projection(condition)
        condition = F.relu(condition)
        condition = condition.view(-1, 1, 16, 16)

        conditioned_enc4 = torch.cat((enc4, condition), dim=1)

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

        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        # NOTE #######################@##
        # adjust this line after to be config arg!!!!
        test_X, test_Y, test_condition = self.dataset.get_validation_sample(10)
        #######################@###



        test_X = test_X.to(self.device)
        test_Y = test_Y.to(self.device)
        test_condition = test_condition.to(self.device)
        for epoch in range(epochs):
            epoch_loss = 0
            psnr_train = 0
            ssim_train = 0


            for batch_idx, (images, conditions, targets) in enumerate(self.dataloader):
                images, targets, conditions = images.to(self.device), conditions.to(self.device), targets.to(self.device)
                optimizer.zero_grad()

                output_train = self.forward(images, conditions)

                # wandb.watch(self, log="all")

                loss_train = self.loss(targets, output_train)
                epoch_loss += loss_train.item()

                psnr_train =+ self.PSNR(output_train, targets)
                ssim_train =+ self.SSIM(output_train, targets)
                

                loss_train.backward()
                optimizer.step()



                if batch_idx%10==0:
                    print(f'Epoch {epoch+1}/{epochs} Batch {batch_idx+1} Loss: {loss_train}')

                # use this block to calculate all test set metrics to avoid affecting model

            # generate images from two sets of sketches: holdout and cuhk
            if gen_images is not None: # boolean
                with torch.no_grad():
                    # generate test images
                    for gen_dir in gen_images_input:
                        image_files = os.listdir(gen_dir)
                        # images = [cv2.resize(cv2.imread(os.path.join(gen_images_input, image), cv2.IMREAD_GRAYSCALE), self.input_size) for image in image_files]
                        image_files = [image for image in image_files if image.endswith('.jpg')]
                        images = [cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(gen_dir, image)), cv2.COLOR_BGR2GRAY), self.input_size) for image in image_files]
                        images = np.array(images)
                        images = torch.from_numpy(images).float().view(-1, 1, self.input_size[0], self.input_size[1])
                        # images = self.dataset.transforms['normalize_sketch'](images)
                        images = images.to(self.device)
                        
                        if gen_condition:
                            conditions = np.load(gen_condition)
                            conditions = torch.from_numpy(conditions).float().view(-1, 15).to(self.device)
                        else:
                            conditions = torch.randint(0, 2, (images.shape[0], 1)).float().to(self.device)
                        output_images = self.forward(images, conditions)
                        output_images = output_images.cpu().numpy()

                        output_dir = gen_images_output + f'/epoch_{epoch}'
                        if not os.path.exists(output_dir):
                                os.mkdir(output_dir)

                        for i, image in enumerate(output_images):
                            try:
                                image = np.transpose(image, (1, 2, 0))
                                cv2.imwrite(output_dir + f'/{i}.jpg', image)
                            except:
                                print('error saving image')
                                continue
                        
                        wandb.log({f"{gen_dir.split('/')[-1]}_epoch_{epoch}_ex": [wandb.Image(np.transpose(img, (1, 2, 0))) for img in output_images]})
                        
                    
                epoch_loss = epoch_loss / len(self.dataloader)
                psnr_train = psnr_train / len(self.dataloader)
                ssim_train = ssim_train / len(self.dataloader)


                print(f'Epoch {epoch} Loss: {epoch_loss}')

            with torch.no_grad():
                output_valid = self.forward(test_X, test_condition)
                loss_valid = self.loss(test_Y, output_valid)
                print(f'Validation Loss: {loss_valid}')

                psnr_valid = self.PSNR(output_valid, test_Y)
                ssim_valid = self.SSIM(output_valid, test_Y)
            self.save_model(save_path)

            
            # mean_epoch_loss_train = sum(epoch_loss_train)/len(epoch_loss_train)
            # mean_epoch_psnr_train = sum(epoch_psnr_train)/len(epoch_psnr_train)
            # mean_epoch_ssim_train = sum(epoch_ssim_train)/len(epoch_ssim_train)

            wandb.log({
                "epoch": epoch+1,
                "train-reconstruction-loss": epoch_loss,
                'valid-reconstruction-loss': loss_valid,
                "train-PSNR": psnr_train,
                'valid-PSNR': psnr_valid,
                'train-SSIM': ssim_train,
                'valid-SSIM': ssim_valid
             })
            
        wandb.log_artifact(self.model_artifact)
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
    'num_img': 200,
    'train_split': 0.8,
    }

    input_size = [128, 128]
    output_size = [128, 128]

    AE = AutoEncoder(input_size=input_size,
        output_size=output_size,
        dataset_info=dataset_info
    )

    AE.train(
        epochs=3, 
        batch_size=128, 
        save_path='model.pth',
        gen_images=True,
        gen_images_input=['generation_input/sketches_holdout',
                          'generation_input/CUHK_sketches'],
        gen_images_output='generation_output'
        )

