import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
import numpy as np
import cv2
from torch.utils.data import DataLoader
import os
import pprint

from celeba import CelebADataset

import wandb
import yaml





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

        self.model_save_path = config['model_save_path']

        # Load configuration information
        self.data_source_config = config['data_source_config']
        self.data_config = config['data_config']
        self.wandb_config = config['wandb_config']
        self.sweep_config = config['sweep_config']
        self.image_gen_config = config['image_gen_config']


        
        # Load input size, output size, and projection size information
        self.input_size = self.data_config['input_size'][0]
        self.output_size = self.data_config['output_size'][0]
        self.projection_size = self.data_config['projection_size']

        # Load CelebADataset object
        self.dataset = CelebADataset(config)



    
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
            for file_dir in self.image_gen_config['gen_image_paths']:

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
                if self.image_gen_config['gen_with_condition']:
                    conditions = self.dataset.condition_data
                    conditions = torch.from_numpy(conditions).float().view(-1, self.data_config['condition_size']).to(self.device)
                else:
                    conditions = torch.randint(0, 2, (prepped_images.shape[0], self.data_config['condition_size'])).float().to(self.device)

                # generate the output images using the trained model and move them to the CPU
                output_images = self.forward(prepped_images, conditions)
                output_images = output_images.cpu().numpy()

                # name the output folder based on the input folder and epoch number, then log the output images to WandB
                gen_output_path = f"{file_dir.split('/')[-1]}_epoch_{epoch+1}_ex"
                wandb.log({gen_output_path: [wandb.Image(np.transpose(img, (1, 2, 0))) for img in output_images]})
        
    def get_optimizer(self, optimizer, learning_rate):

        if optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(),
                              lr=learning_rate, momentum=0.9)
        elif optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(),
                                lr=learning_rate)
        elif optimizer == "adamw":
            optimizer = torch.optim.Adam(self.parameters(),
                                lr=learning_rate)
        return optimizer
    

    def train(self, config=None):
        """
        Trains the autoencoder model on the specified dataset for the specified number of epochs,
        logging loss and evaluation metrics to Weights and Biases.
        """
        wandb_config = self.wandb_config
        sweep_config = self.sweep_config

        model_name = wandb_config['model_name']
        dataset_name = wandb_config['dataset_name']

        # Initialize Weights and Biases artifact for model
        model_artifact = wandb.Artifact(model_name, type="model")

        # Load dataset artifact and dataset information
        dataset_artifact = wandb.Artifact(dataset_name, "dataset")
        dataset_artifact.add_dir("model_input/sketches/")
        # wandb.use_artifact(dataset_artifact)

        with wandb.init(config=sweep_config):

            config = wandb.config
            
            #optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            optimizer = self.get_optimizer(optimizer = config.optimizer, 
                                           learning_rate = config.learning_rate)

            self.dataloader = DataLoader(self.dataset, 
                                         batch_size=config.batch_size, 
                                         shuffle=True)

            # Get a validation sample from the dataset for evaluation
            test_X, test_Y, test_condition = self.dataset.get_validation_sample(self.data_config['validation_sample_size'])

            # Convert validation data to device (CPU or GPU)
            test_X = test_X.to(self.device)
            test_Y = test_Y.to(self.device)
            test_condition = test_condition.to(self.device)

            # Loop through each epoch
            for epoch in range(config.epochs):
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
                if self.image_gen_config['gen_image_paths'] is not None:
                    self.genImages(epoch)

                

                # Log metrics to Weights and Biases
                wandb.log({
                    "epoch": epoch,
                    "train-reconstruction-loss": epoch_loss,
                    'valid-reconstruction-loss': loss_valid,

                    "train-PSNR": psnr_train,
                    'valid-PSNR': psnr_valid,
                    'train-SSIM': ssim_train,
                    'valid-SSIM': ssim_valid
                    },)

            # Save the model
            self.save_model(self.model_save_path)
            wandb.log_artifact(model_artifact)

        


    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))

if __name__ == '__main__':
    # read config
    with open('config.yaml', "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    wandb_config = config['wandb_config']
    sweep_config = config['sweep_config']

    

    pprint.pprint(sweep_config)

    # Define model name and dataset name for Weights and Biases
    project_name = wandb_config['project_name']
    

    # establish sweep id
    sweep_id = wandb.sweep(sweep_config, project=project_name)

    # instantiate AutoEncoder
    AE = AutoEncoder(config)

    # train AutoEncoder
    wandb.agent(sweep_id, AE.train, count=5)

    # Save the trained model as an artifact in Weights and Biases
    

    # End the Weights and Biases session
    wandb.finish()

