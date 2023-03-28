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
    """
    A class containing building blocks for the U-Net architecture.

    Methods
    -------
    conv_block(in_channels, out_channels)
        Returns a convolutional block consisting of two convolutional layers with ReLU activation.
    upconv_block(in_channels, out_channels)
        Returns an up-convolutional block that performs transpose convolution with ReLU activation.
    """
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
    def __init__(self, config):
        """Instantiate an AutoEncoder object.

        Args:
            config (dict): A dictionary containing configuration information for the autoencoder.

        Raises:
            ValueError: If the configuration information is invalid.
        """
        super(AutoEncoder, self).__init__()

        # Determine whether to use a GPU or CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Load configuration information
        self.data_source_info = config['data_source_info']
        self.data_info = config['data_info']
        self.training_info = config['training_info']
        self.wandb_info = config['wandb_info']
        self.image_gen_info = config['image_gen_info']

        # Define model name and dataset name for Weights and Biases
        model_name = self.wandb_info['model_name']
        dataset_name = self.wandb_info['dataset']

        # Initialize Weights and Biases artifact for model
        self.model_artifact = wandb.Artifact(model_name, type="model")

        

        # Load hyperparameters for training
        self.learning_rate = self.training_info['learning_rate']
        self.batch_size = self.training_info['batch_size']
        self.epochs = self.training_info['epochs']



        # Initialize Weights and Biases run
        wandb.init(
            project=model_name,
            config={
                "learning_rate": self.learning_rate,
                "architecture": self.wandb_info['architecture'],
                "dataset": self.wandb_info['dataset'],
                'epochs': self.epochs,
                'batch_size': self.batch_size,
            }
        )

        # Load dataset artifact and dataset information
        dataset_artifact = wandb.Artifact(dataset_name, "dataset")
        dataset_artifact.add_dir("model_input/sketches/")
        dataset = wandb.use_artifact(dataset_artifact)

        
        # Load input size, output size, and projection size information
        self.input_size = self.data_info['input_size'][0]
        self.output_size = self.data_info['output_size'][0]
        self.projection_size = self.data_info['projection_size']

        # Load CelebADataset object
        self.dataset = CelebADataset(config)

        # Define encoder layers
        self.encoder_conv1 = unetBlocks.conv_block(1, 32)
        self.encoder_conv2 = unetBlocks.conv_block(32, 64)
        self.encoder_conv3 = unetBlocks.conv_block(64, 128)
        self.encoder_conv4 = unetBlocks.conv_block(128, 256)

        self.max_pool = nn.MaxPool2d(2)

        self.condition_projection = nn.Linear(self.data_info['condition_size'], self.projection_size)

        # Define decoder layers
        self.decoder_conv1 = unetBlocks.conv_block(256, 128)
        self.decoder_conv2 = unetBlocks.conv_block(128, 64)
        self.decoder_conv3 = unetBlocks.conv_block(64, 32)

        self.decoder_upconv1 = unetBlocks.upconv_block(257, 128)
        self.decoder_upconv2 = unetBlocks.upconv_block(128, 64)
        self.decoder_upconv3 = unetBlocks.upconv_block(64, 32)

        # Define output layer
        self.output_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        self.to(self.device)

    
    def forward(self, input, condition):
        """
        Forward pass of the AutoEncoder.

        Args:
            input: Input tensor of size (batch_size, 1, input_size, input_size)
            condition: Conditional tensor of size (batch_size, condition_size)

        Returns:
            out: Output tensor of size (batch_size, 3, output_size, output_size)
        """

        # Encoder layers
        enc1 = self.encoder_conv1(input)
        enc2 = self.encoder_conv2(self.max_pool(enc1))
        enc3 = self.encoder_conv3(self.max_pool(enc2))
        enc4 = self.encoder_conv4(self.max_pool(enc3))

        # Projecting the conditional data and reshaping it to match the dimensions of the encoder features
        condition = self.condition_projection(condition)
        condition = F.relu(condition)
        condition = condition.view(-1, 1, 16, 16)

        # Concatenating the conditioned vector with the output of the final encoder layer
        conditioned_enc4 = torch.cat((enc4, condition), dim=1)

        # Decoder layers
        dec1 = self.decoder_upconv1(conditioned_enc4)
        dec1 = torch.cat((enc3, dec1), dim=1)
        dec1 = self.decoder_conv1(dec1)

        dec2 = self.decoder_upconv2(dec1)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder_conv2(dec2)

        dec3 = self.decoder_upconv3(dec2)
        dec3 = torch.cat((enc1, dec3), dim=1)
        dec3 = self.decoder_conv3(dec3)

        # Output
        out = self.output_conv(dec3)

        return out

    
    def loss(self, ground_truth, output):
        """
        Computes the Mean Squared Error (MSE) loss between the ground truth images and the predicted output images.

        Args:
        - ground_truth (torch.Tensor): a tensor representing the ground truth images
        - output (torch.Tensor): a tensor representing the predicted output images

        Returns:
        - loss (torch.Tensor): the MSE loss between the ground truth and predicted output images
        """
        loss = F.mse_loss(ground_truth, output)
        return loss

    def PSNR(self, output, ground_truth):
        """
        Computes the Peak Signal-to-Noise Ratio (PSNR) between the ground truth images and the predicted output images.

        Args:
        - output (torch.Tensor): a tensor representing the predicted output images
        - ground_truth (torch.Tensor): a tensor representing the ground truth images

        Returns:
        - psnr (torch.Tensor): the PSNR between the ground truth and predicted output images
        """
        psnr = peak_signal_noise_ratio(output, ground_truth)
        return psnr
    
    def SSIM(self, output, ground_truth):
        """
        Computes the Structural Similarity Index Measure (SSIM) between the ground truth images and the predicted output images.

        Args:
        - output (torch.Tensor): a tensor representing the predicted output images
        - ground_truth (torch.Tensor): a tensor representing the ground truth images

        Returns:
        - ssim (torch.Tensor): the SSIM between the ground truth and predicted output images
        """
        ssim = structural_similarity_index_measure(output, ground_truth)
        return ssim
    
    def genImages(self, epoch):
        """Generates images during training and logs them to WandB.

        Args:
            epoch (int): The current epoch number.
        """
        # Disable gradient calculation to speed up image generation
        with torch.no_grad():

            # generate test images for each path in `gen_image_paths`
            for file_dir in self.image_gen_info['gen_image_paths']:

                # get the list of image files in the directory
                image_files = os.listdir(file_dir)
                image_files = [image for image in image_files if image.endswith('.jpg')]

                # pre-process the images
                prepped_images = []
                for img_file in image_files:
                    img = cv2.imread(os.path.join(file_dir, img_file))
                    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    prep_img = cv2.resize(img_color, (self.input_size, self.input_size))
                    prepped_images.append(prep_img)

                # convert the pre-processed images to tensors and move them to the device (GPU/CPU)
                prepped_images = np.array(prepped_images)
                prepped_images = torch.from_numpy(prepped_images).float().view(-1, 1, self.input_size, self.input_size)
                prepped_images = prepped_images.to(self.device)

                # get the conditions to generate images
                if self.image_gen_info['gen_with_condition']:
                    conditions = self.dataset.condition_data
                    conditions = torch.from_numpy(conditions).float().view(-1, self.data_info['condition_size']).to(self.device)
                else:
                    conditions = torch.randint(0, 2, (prepped_images.shape[0], self.data_info['condition_size'])).float().to(self.device)

                # generate the output images using the trained model and move them to the CPU
                output_images = self.forward(prepped_images, conditions)
                output_images = output_images.cpu().numpy()

                # name the output folder based on the input folder and epoch number, then log the output images to WandB
                gen_output_path = f"{file_dir.split('/')[-1]}_epoch_{epoch+1}_ex"
                wandb.log({gen_output_path: [wandb.Image(np.transpose(img, (1, 2, 0))) for img in output_images]})
        

    def train(self):
        """
        Trains the autoencoder model on the specified dataset for the specified number of epochs,
        logging loss and evaluation metrics to Weights and Biases.
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # Get a validation sample from the dataset for evaluation
        test_X, test_Y, test_condition = self.dataset.get_validation_sample(self.data_info['validation_sample_size'])

        # Convert validation data to device (CPU or GPU)
        test_X = test_X.to(self.device)
        test_Y = test_Y.to(self.device)
        test_condition = test_condition.to(self.device)

        # Loop through each epoch
        for epoch in range(self.epochs):
            epoch_loss = 0
            psnr_train = 0
            ssim_train = 0

            # Loop through each batch in the training data
            for batch_idx, (images, conditions, targets) in enumerate(self.dataloader):
                images, targets, conditions = images.to(self.device), conditions.to(self.device), targets.to(self.device)
                optimizer.zero_grad()

                # Generate output from the autoencoder
                output_train = self.forward(images, conditions)

                # Compute the loss
                loss_train = self.loss(targets, output_train)
                epoch_loss += loss_train.item()

                # Compute evaluation metrics (PSNR and SSIM)
                psnr_train =+ self.PSNR(output_train, targets)
                ssim_train =+ self.SSIM(output_train, targets)

                # Compute gradients and update weights
                loss_train.backward()
                optimizer.step()

            # Compute average loss for the epoch
            epoch_loss = epoch_loss / len(self.dataloader)
            psnr_train = psnr_train / len(self.dataloader)
            ssim_train = ssim_train / len(self.dataloader)

            print(f'Epoch {epoch+1} Training Loss:   {epoch_loss}')

            # calculate validation loss and metrics
            
            with torch.no_grad():
                output_valid = self.forward(test_X, test_condition)
                loss_valid = self.loss(test_Y, output_valid)
                print(f'Epoch {epoch+1} Validation Loss: {loss_valid}\n')

                psnr_valid = self.PSNR(output_valid, test_Y)
                ssim_valid = self.SSIM(output_valid, test_Y)

            # Generate images if specified in configuration
            if self.image_gen_info['gen_image_paths'] is not None:
                self.genImages(epoch)

            # Save the model
            self.save_model(self.training_info['save_path'])

            # Log metrics to Weights and Biases
            wandb.log({
                "epoch": epoch + 1,
                "train-reconstruction-loss": epoch_loss,
                'valid-reconstruction-loss': loss_valid,

                "train-PSNR": psnr_train,
                'valid-PSNR': psnr_valid,
                'train-SSIM': ssim_train,
                'valid-SSIM': ssim_valid
                },
                step=epoch 
            )

        # Save the trained model as an artifact in Weights and Biases
        wandb.log_artifact(self.model_artifact)

        # End the Weights and Biases session
        wandb.finish()


    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))

if __name__ == '__main__':
    # read config
    with open('config.yaml', "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # instantiate AutoEncoder
    AE = AutoEncoder(config)

    # train AutoEncoder
    AE.train()

