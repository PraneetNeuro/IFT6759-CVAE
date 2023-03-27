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
import yaml

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
    def __init__(self, config
            # self, input_size, output_size, num_img=None, dataset_info={
    #     'source': 'sketches',
    #     'target': 'original',
    #     'condition': 'conditions.npy',
    #     'num_img': 29999,
    #     'train_split': 0.9999,
    # }, model_config={
    #     'condition': True,
    #     'condition_type': 'mul',
    #     'skip_connection': True,
    # }
        ):
        super(AutoEncoder, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')


        self.data_source_info = config['data_source_info']
        self.data_info = config['data_info']
        self.training_info = config['training_info']
        self.wandb_info = config['wandb_info']
        self.image_gen_info = config['image_gen_info']

        # first arg denotes model version
        self.model_artifact = wandb.Artifact("baseline_test_versioning", type="model")

        self.learning_rate = self.training_info['learning_rate']
        self.batch_size = self.training_info['batch_size']
        self.epochs = self.training_info['epochs']

        
        #  WANDB INITIALIZATION
        wandb.init(
            # set the wandb project where this run will be logged
            project=self.wandb_info['model_name'],
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": self.learning_rate,
            "architecture": self.wandb_info['architecture'],
            "dataset": self.wandb_info['dataset'],
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            
            }
        )
        dataset_artifact = wandb.Artifact(self.wandb_info['dataset'], "dataset")
        dataset_artifact.add_dir("model_input/sketches/")
        dataset = wandb.use_artifact(dataset_artifact)

        self.input_size = self.data_info['input_size'][0]
        self.output_size = self.data_info['output_size'][0]
        self.projection_size = self.data_info['projection_size']
        # self.model_config = model_config

        self.dataset = CelebADataset(config)


        # Encoder layers
        self.encoder_conv1 = unetBlocks.conv_block(1, 32)
        self.encoder_conv2 = unetBlocks.conv_block(32, 64)
        self.encoder_conv3 = unetBlocks.conv_block(64, 128)
        self.encoder_conv4 = unetBlocks.conv_block(128, 256)

        self.max_pool = nn.MaxPool2d(2)

        self.condition_projection = nn.Linear(self.data_info['condition_size'], self.projection_size)
        
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
    # def train(self, epochs, batch_size, save_path, gen_images = None, gen_condition = None, gen_images_input = None, gen_images_output = None
    def train(self, ):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # NOTE #######################@##
        # adjust this line after to be config arg!!!!
        test_X, test_Y, test_condition = self.dataset.get_validation_sample(self.data_info['validation_sample_size'])
        #######################@###

        self.gen_image_paths = self.image_gen_info['gen_image_paths']
        self.gen_with_condition = self.image_gen_info['gen_with_condition']

        test_X = test_X.to(self.device)
        test_Y = test_Y.to(self.device)
        test_condition = test_condition.to(self.device)
        for epoch in range(self.epochs):
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




            # generate images from two sets of sketches: holdout and cuhk
            if self.gen_image_paths is not None: # boolean
                with torch.no_grad():

                    # generate test images
                    for file_dir in self.gen_image_paths:
                        image_files = os.listdir(file_dir)
                        # images = [cv2.resize(cv2.imread(os.path.join(gen_images_input, image), cv2.IMREAD_GRAYSCALE), self.input_size) for image in image_files]
                        image_files = [image for image in image_files if image.endswith('.jpg')]

                        prepped_images = []
                        for img_file in image_files:
                            img = cv2.imread(os.path.join(file_dir, img_file))
                            img_color = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            prep_img = cv2.resize(img_color, (self.input_size, self.input_size))
                            prepped_images.append(prep_img)

                        prepped_images = np.array(prepped_images)
                        prepped_images = torch.from_numpy(prepped_images).float().view(-1, 1, self.input_size, self.input_size)
                        # images = self.dataset.transforms['normalize_sketch'](images)
                        prepped_images = prepped_images.to(self.device)
                        
                        # need to integrate with num_img arg
                        if self.gen_with_condition:
                            # conditions = np.load(self.data_source_info['condition_path'])
                            conditions = self.dataset.condition_data
                            conditions = torch.from_numpy(conditions).float().view(-1, self.data_info['condition_size']).to(self.device)
                        else:
                            conditions = torch.randint(0, 2, (prepped_images.shape[0], self.data_info['condition_size'])).float().to(self.device)
                        output_images = self.forward(prepped_images, conditions)
                        output_images = output_images.cpu().numpy()

                        gen_output_path = f"{file_dir.split('/')[-1]}_epoch_{epoch+1}_ex"
                        wandb.log({gen_output_path: [wandb.Image(np.transpose(img, (1, 2, 0))) for img in output_images]})
                        
                    
                epoch_loss = epoch_loss / len(self.dataloader)
                psnr_train = psnr_train / len(self.dataloader)
                ssim_train = ssim_train / len(self.dataloader)


                print(f'Epoch {epoch+1} Training Loss:   {epoch_loss}')

            with torch.no_grad():
                output_valid = self.forward(test_X, test_condition)
                loss_valid = self.loss(test_Y, output_valid)
                print(f'Epoch {epoch+1} Validation Loss: {loss_valid}\n')

                psnr_valid = self.PSNR(output_valid, test_Y)
                ssim_valid = self.SSIM(output_valid, test_Y)
            self.save_model(self.training_info['save_path'])


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
    with open('config.yaml', "r") as yaml_file:
        config = yaml.safe_load(yaml_file)


    AE = AutoEncoder(config)

    AE.train()

