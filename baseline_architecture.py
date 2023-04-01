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

        # Define encoder layers
        self.encoder_conv1 = unetBlocks.conv_block(1, 32)
        self.encoder_conv2 = unetBlocks.conv_block(32, 64)
        self.encoder_conv3 = unetBlocks.conv_block(64, 128)
        self.encoder_conv4 = unetBlocks.conv_block(128, 256)

        self.max_pool = nn.MaxPool2d(2)

        self.condition_projection = nn.Linear(self.data_config['condition_size'], self.projection_size)

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
    
    def getSketches(self, sketch_path):
        """
        Returns a list of sketches from the given sketch path.

        Args:
        - sketch_path (str): the path to the directory containing the sketches

        Returns:
        - sketches (list): a list of sketches
        """

        prepped_sketches = []
        sketch_files = sorted(os.listdir(sketch_path))
        if '.DS_Store' in sketch_files:
            sketch_files.remove('.DS_Store')
        for sketch_file in sketch_files:
            
            sketch = cv2.imread(os.path.join(sketch_path, sketch_file))
            sketch_color = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
            prepped_sketch = cv2.resize(sketch_color, (self.input_size, self.input_size))
            prepped_sketches.append(prepped_sketch)

        # convert the pre-processed images to tensors and move them to the device (GPU/CPU)
        prepped_sketches = np.array(prepped_sketches)
        prepped_sketches = torch.from_numpy(prepped_sketches).float().view(-1, 1, self.input_size, self.input_size)

        return prepped_sketches, sketch_files

  


    
    def genImages(self, epoch):

        """Generates images during training and logs them to WandB.

        Args:
            epoch (int): The current epoch number.
        """

        # read photos


        # get "bad" photos
        bad_photo_path = self.image_gen_config['bad_photo_path']
        bad_photo_files = sorted(os.listdir(bad_photo_path))
        if '.DS_Store' in bad_photo_files:
            bad_photo_files.remove('.DS_Store')
        bad_photos = [cv2.imread(os.path.join(bad_photo_path, img_file))for img_file in bad_photo_files]

        # get CUHK photos
        cuhk_photo_path = self.image_gen_config['cuhk_photo_path']
        cuhk_photo_files = sorted(os.listdir(cuhk_photo_path))
        cuhk_photos = [cv2.imread(os.path.join(cuhk_photo_path, img_file)) for img_file in cuhk_photo_files]

        # get holdout photos
        holdout_photo_path = self.image_gen_config['holdout_photo_path']
        holdout_photo_files = sorted(os.listdir(holdout_photo_path))
        orig_photos = [cv2.imread(os.path.join(holdout_photo_path, img_file)) for img_file in holdout_photo_files]



        # read sketches

        # get holdout sketches
        simple_sketch_path = self.image_gen_config['simple_sketch_path']
        simple_sketches, simple_sketch_files = self.getSketches(simple_sketch_path)
        detail_sketch_path = self.image_gen_config['detail_sketch_path']
        detail_sketches, detail_sketch_files = self.getSketches(detail_sketch_path)


        # get "bad" sketches
        bad_sketch_path = self.image_gen_config['bad_sketch_path']
        bad_sketches, bad_sketch_files = self.getSketches(bad_sketch_path)
        bad_sketches = bad_sketches.to(self.device)

        # get CUHK sketches
        cuhk_sketch_path = self.image_gen_config['cuhk_sketch_path']
        cuhk_sketches, cuhk_sketch_files = self.getSketches(cuhk_sketch_path)

            

        # get the condition vectors to generate images
        condition_size = self.data_config['condition_size']
        if self.image_gen_config['gen_with_condition']:
            conditions = self.dataset.condition_data
            conditions = torch.from_numpy(conditions).float().view(-1, condition_size).to(self.device)
        else:
            bad_sketch_conditions    = torch.zeros(bad_sketches.shape[0], condition_size).float().to(self.device)
            simple_sketch_conditions = torch.zeros(simple_sketches.shape[0], condition_size).float().to(self.device)
            detail_sketch_conditions = torch.zeros(detail_sketches.shape[0], condition_size).float().to(self.device)
            cuhk_sketch_conditions   = torch.zeros(cuhk_sketches.shape[0], condition_size).float().to(self.device)

        # generate the output images using the trained model and move them to the CPU
        with torch.no_grad():
            bad_sketch_output = self.forward(bad_sketches, bad_sketch_conditions).cpu().numpy()
            # bad_sketch_output = bad_sketch_output.cpu().numpy()

            simple_sketch_output = self.forward(simple_sketches, simple_sketch_conditions).cpu().numpy()
            detail_sketch_output = self.forward(detail_sketches, detail_sketch_conditions).cpu().numpy()
            cuhk_sketch_output = self.forward(cuhk_sketches, cuhk_sketch_conditions).cpu().numpy()
        

        columns = ['id', 'sketch', 'photo', 'generation']
        
        # bad sketch table
        bad_sketch_table = wandb.Table(columns=columns)
        for name, sketch, photo, generation in zip(bad_sketch_files, bad_sketches, bad_photos, bad_sketch_output):
            _ = sketch
            sketch = wandb.Image(sketch)

            photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
            photo = wandb.Image(photo)
            # [wandb.Image(cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_BGR2RGB)) for img in output_images]})

            generation = np.transpose(generation, (1, 2, 0))
            generation = cv2.cvtColor(generation, cv2.COLOR_BGR2RGB)
            generation = wandb.Image(generation)


            bad_sketch_table.add_data(name, sketch, photo, generation)
        wandb.log({f"bad_sketches_epoch_{epoch+1}_{str(wandb.run.id)}": bad_sketch_table})

        # CUHK sketch table
        cuhk_sketch_table = wandb.Table(columns=columns)
        for name, sketch, photo, generation in zip(cuhk_sketch_files, cuhk_sketches, cuhk_photos, cuhk_sketch_output):
            _ = sketch
            sketch = wandb.Image(sketch)

            photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
            photo = wandb.Image(photo)
            # [wandb.Image(cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_BGR2RGB)) for img in output_images]})

            generation = np.transpose(generation, (1, 2, 0))
            generation = cv2.cvtColor(generation, cv2.COLOR_BGR2RGB)
            generation = wandb.Image(generation)


            cuhk_sketch_table.add_data(name, sketch, photo, generation)
        wandb.log({f"cuhk_sketches_epoch_{epoch+1}_{str(wandb.run.id)}": cuhk_sketch_table})

        # simple sketch table
        simple_sketch_table = wandb.Table(columns=columns)
        for name, sketch, photo, generation in zip(simple_sketch_files, simple_sketches, orig_photos, simple_sketch_output):
            _ = sketch
            sketch = wandb.Image(sketch)

            photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
            photo = wandb.Image(photo)
            # [wandb.Image(cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_BGR2RGB)) for img in output_images]})

            generation = np.transpose(generation, (1, 2, 0))
            generation = cv2.cvtColor(generation, cv2.COLOR_BGR2RGB)
            generation = wandb.Image(generation)


            simple_sketch_table.add_data(name, sketch, photo, generation)
        wandb.log({f"simple_sketches_epoch_{epoch+1}_{str(wandb.run.id)}": simple_sketch_table})

        # simple sketch table
        detail_sketch_table = wandb.Table(columns=columns)
        for name, sketch, photo, generation in zip(detail_sketch_files, detail_sketches, orig_photos, detail_sketch_output):
            _ = sketch
            sketch = wandb.Image(sketch)

            photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
            photo = wandb.Image(photo)
            # [wandb.Image(cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_BGR2RGB)) for img in output_images]})

            generation = np.transpose(generation, (1, 2, 0))
            generation = cv2.cvtColor(generation, cv2.COLOR_BGR2RGB)
            generation = wandb.Image(generation)


            detail_sketch_table.add_data(name, sketch, photo, generation)
        wandb.log({f"detil_sketches_epoch_{epoch+1}_{str(wandb.run.id)}": detail_sketch_table})



        # name the output folder based on the input folder and epoch number, then log the output images to WandB
        # simple_sketch_label = f"{simple_sketch_path.split('/')[-1]}_epoch_{epoch+1}"
        # detail_sketch_label = f"{detail_sketch_path.split('/')[-1]}_epoch_{epoch+1}"

        
        # create a table for logging to wandb
        

        # simple_sketch_art = wandb.Artifact("simple_sketches" + str(wandb.run.id), type="predictions")
        # holdout_table = wandb.Table(columns=columns)
        # for file, sketch, photo, generation in zip(simple_files, simple_sketches, orig_photos, simple_sketch_output):
        #     generation = wandb.Image(np.transpose(generation, (1, 2, 0))) 
        #     holdout_table.add_data(file, sketch, photo, generation)

        # simple_sketch_art.add(holdout_table, "predictions")
        # wandb.run.log_artifact(simple_sketch_art)    

        
        
        # bad_sketch_label = f"{bad_sketch_path.split('/')[-1]}_epoch_{epoch+1}"
        # cuhk_sketch_label = f"{cuhk_sketch_path.split('/')[-1]}_epoch_{epoch+1}"


        # generations = [for img in output_images]
        # wandb.log({gen_output_path: generations})



            
        
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
        elif optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(),
                                lr=learning_rate)
        else:
            raise ValueError('Optimizer not supported')
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
                if self.image_gen_config['gen_img'] is not None:
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
    wandb.agent(sweep_id, AE.train, count=1)

    # Save the trained model as an artifact in Weights and Biases
    

    # End the Weights and Biases session
    wandb.finish()

