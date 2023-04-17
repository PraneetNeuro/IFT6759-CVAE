import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from ignite.metrics import FID, InceptionScore

import torchvision
import torchvision.models as models
import numpy as np
import cv2
from torch.utils.data import DataLoader
import os
import pprint

from celeba import CelebADataset

import wandb
import yaml
    
class PerceptualLoss(nn.Module):
    def __init__(self, layers=[3, 8, 15, 22, 29]):
        super(PerceptualLoss, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True).features
        self.criterion = nn.MSELoss()
        self.layers = layers
        for param in self.vgg16.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_vgg, y_vgg = x, y
        loss = 0
        for i in range(max(self.layers) + 1):
            x_vgg = self.vgg16[i](x_vgg)
            y_vgg = self.vgg16[i](y_vgg)
            if i in self.layers:
                loss += self.criterion(x_vgg, y_vgg.detach())
        return loss


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(UNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.block(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(UNetUpBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.block(x)


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
            self.device = torch.device('mps')

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

        # Custom loss functions
        self.perceptual_loss = PerceptualLoss()
        
        # Encoder layers
        self.enc1 = UNetBlock(1, 32)
        self.enc2 = UNetBlock(32, 64)
        self.enc3 = UNetBlock(64, 128)
        self.enc4 = UNetBlock(128, 256)

        # Bottleneck layer
        self.center = UNetBlock(256, 512)

        # Decoder layers
        self.up4 = UNetUpBlock(512, 256)
        self.dec4 = UNetBlock(256 + 256, 256) 
        self.up3 = UNetUpBlock(256, 128)
        self.dec3 = UNetBlock(128 + 128, 128)  
        self.up2 = UNetUpBlock(128, 64)
        self.dec2 = UNetBlock(64 + 64, 64)     
        self.up1 = UNetUpBlock(64, 32)
        self.dec1 = UNetBlock(32 + 32, 32)

        # Output layer
        self.output_conv = nn.Conv2d(32, 3, kernel_size=1)

        # Pooling layer for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.PSNR = PeakSignalNoiseRatio()
        self.SSIM = StructuralSimilarityIndexMeasure()
        self.FID = FrechetInceptionDistance(normalize=True)
        self.IS = InceptionScore()

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

        # Encoder
        enc1_out = self.enc1(input)
        enc2_out = self.enc2(self.pool(enc1_out))
        enc3_out = self.enc3(self.pool(enc2_out))
        enc4_out = self.enc4(self.pool(enc3_out))

        # Center
        center_out = self.center(self.pool(enc4_out))
        
        # Decoder
        up4_out = self.up4(center_out)
        up4_out = torch.cat((up4_out, enc4_out), dim=1)  # Skip connection
        dec4_out = self.dec4(up4_out)
        up3_out = self.up3(dec4_out)
        up3_out = torch.cat((up3_out, enc3_out), dim=1)  # Skip connection
        dec3_out = self.dec3(up3_out)
        up2_out = self.up2(dec3_out)
        up2_out = torch.cat((up2_out, enc2_out), dim=1)  # Skip connection
        dec2_out = self.dec2(up2_out)
        up1_out = self.up1(dec2_out)
        up1_out = torch.cat((up1_out, enc1_out), dim=1)  # Skip connection
        dec1_out = self.dec1(up1_out)

        # Output
        output = self.output_conv(dec1_out)
        output = torch.sigmoid(output)
        return output

    
    def loss(self, ground_truth, output, custom_loss=True):
        """
        Computes the Mean Squared Error (MSE) loss between the ground truth images and the predicted output images.

        Args:
        - ground_truth (torch.Tensor): a tensor representing the ground truth images
        - output (torch.Tensor): a tensor representing the predicted output images

        Returns:
        - loss (torch.Tensor): the MSE loss between the ground truth and predicted output images
        """
        if custom_loss:
            loss = self.perceptual_loss(output, ground_truth)
        else:
            loss = F.mse_loss(ground_truth, output)
        return loss

    def calculate_metrics(self, holdout_outputs, holdout_target):
        """
        Compute four metrics for validation data like hold-outs and test set. DO NOT use for training data
        or data divided in batches.
        :param holdout_outputs: generated images
        :param holdout_target: originals
        :return: ssim_score, fid_score, incept_score and psnr_score
        """
        # Initialise
        ssim = StructuralSimilarityIndexMeasure()
        fid = FrechetInceptionDistance(normalize=True)
        incept = InceptionScore()
        psnr = PeakSignalNoiseRatio()

        ssim.update(holdout_outputs, holdout_target)
        fid.update(holdout_outputs, real=False)
        fid.update(holdout_target, real=True)
        incept.update(holdout_outputs)
        psnr.update(holdout_outputs, holdout_target)

        ssim_score, fid_score, incept_score, psnr_score = ssim.compute(), fid.compute(), incept.compute(), psnr.compute()
        return ssim_score, fid_score, incept_score, psnr_score
    
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
        simple_sketches = simple_sketches.to(self.device)

        detail_sketch_path = self.image_gen_config['detail_sketch_path']
        detail_sketches, detail_sketch_files = self.getSketches(detail_sketch_path)
        detail_sketches = detail_sketches.to(self.device)


        # get "bad" sketches
        bad_sketch_path = self.image_gen_config['bad_sketch_path']
        bad_sketches, bad_sketch_files = self.getSketches(bad_sketch_path)
        bad_sketches = bad_sketches.to(self.device)

        # get CUHK sketches
        cuhk_sketch_path = self.image_gen_config['cuhk_sketch_path']
        cuhk_sketches, cuhk_sketch_files = self.getSketches(cuhk_sketch_path)
        cuhk_sketches = cuhk_sketches.to(self.device)

            

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
        
        # create image tables

        columns = ['id', 'sketch', 'photo', 'generation', 'ssim', 'inception_score', 'fid']

        # bad sketch table
        bad_sketch_table = wandb.Table(columns=columns)
        for name, sketch, photo, generation in zip(bad_sketch_files, bad_sketches, bad_photos, bad_sketch_output):
            _ = sketch
            # convert photo to tensor and proper size
            metric_photo = torch.tensor(np.transpose(photo, (2, 0, 1)))
            metric_photo = torchvision.transforms.Resize((self.output_size, self.output_size), antialias=True)(metric_photo).to(torch.float32)
            metric_photo = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(metric_photo)
            metric_photo = metric_photo.unsqueeze(0)
            metric_gen = torch.tensor(generation).unsqueeze(0)

            ssim_score, fid_score, inception_score, psnr_score = self.calculate_metrics(metric_gen, metric_photo)

            sketch = wandb.Image(sketch)

            photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
            photo = wandb.Image(photo)
            # [wandb.Image(cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_BGR2RGB)) for img in output_images]})

            generation = np.transpose(generation, (1, 2, 0))
            generation = cv2.cvtColor(generation, cv2.COLOR_BGR2RGB)
            generation = wandb.Image(generation)

            bad_sketch_table.add_data(name, sketch, photo, generation, ssim_score, inception_score, fid_score)
        wandb.log({f"bad_sketches_epoch_{epoch+1}_{str(wandb.run.id)}": bad_sketch_table})

        # CUHK sketch table
        cuhk_sketch_table = wandb.Table(columns=columns)
        for name, sketch, photo, generation in zip(cuhk_sketch_files, cuhk_sketches, cuhk_photos, cuhk_sketch_output):
            _ = sketch

            metric_photo = torch.tensor(np.transpose(photo, (2, 0, 1)))
            metric_photo = torchvision.transforms.Resize((self.output_size, self.output_size), antialias=True)(metric_photo).to(torch.float32)
            metric_photo = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(metric_photo)
            metric_photo = metric_photo.unsqueeze(0)
            metric_gen = torch.tensor(generation).unsqueeze(0)

            ssim_score, fid_score, inception_score, psnr_score = self.calculate_metrics(metric_gen, metric_photo)

            sketch = wandb.Image(sketch)

            photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
            photo = wandb.Image(photo)
            # [wandb.Image(cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_BGR2RGB)) for img in output_images]})

            generation = np.transpose(generation, (1, 2, 0))
            generation = cv2.cvtColor(generation, cv2.COLOR_BGR2RGB)
            generation = wandb.Image(generation)

            cuhk_sketch_table.add_data(name, sketch, photo, generation, ssim_score, inception_score, fid_score)
        wandb.log({f"cuhk_sketches_epoch_{epoch+1}_{str(wandb.run.id)}": cuhk_sketch_table})

        # simple sketch table
        simple_sketch_table = wandb.Table(columns=columns)
        for name, sketch, photo, generation in zip(simple_sketch_files, simple_sketches, orig_photos, simple_sketch_output):
            _ = sketch

            metric_photo = torch.tensor(np.transpose(photo, (2, 0, 1)))
            metric_photo = torchvision.transforms.Resize((self.output_size, self.output_size), antialias=True)(metric_photo).to(torch.float32)
            metric_photo = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(metric_photo)
            metric_photo = metric_photo.unsqueeze(0)
            metric_gen = torch.tensor(generation).unsqueeze(0)

            ssim_score, fid_score, inception_score, psnr_score = self.calculate_metrics(metric_gen, metric_photo)

            sketch = wandb.Image(sketch)

            photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
            photo = wandb.Image(photo)
            # [wandb.Image(cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_BGR2RGB)) for img in output_images]})

            generation = np.transpose(generation, (1, 2, 0))
            generation = cv2.cvtColor(generation, cv2.COLOR_BGR2RGB)
            generation = wandb.Image(generation)


            simple_sketch_table.add_data(name, sketch, photo, generation, ssim_score, inception_score, fid_score)
        wandb.log({f"simple_sketches_epoch_{epoch+1}_{str(wandb.run.id)}": simple_sketch_table})

        # simple sketch table
        detail_sketch_table = wandb.Table(columns=columns)
        for name, sketch, photo, generation in zip(detail_sketch_files, detail_sketches, orig_photos, detail_sketch_output):
            _ = sketch

            metric_photo = torch.tensor(np.transpose(photo, (2, 0, 1)))
            metric_photo = torchvision.transforms.Resize((self.output_size, self.output_size), antialias=True)(metric_photo).to(torch.float32)
            metric_photo = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(metric_photo)
            metric_photo = metric_photo.unsqueeze(0)
            metric_gen = torch.tensor(generation).unsqueeze(0)

            ssim_score, fid_score, inception_score, psnr_score = self.calculate_metrics(metric_gen, metric_photo)

            sketch = wandb.Image(sketch)

            photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
            photo = wandb.Image(photo)
            # [wandb.Image(cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_BGR2RGB)) for img in output_images]})

            generation = np.transpose(generation, (1, 2, 0))
            generation = cv2.cvtColor(generation, cv2.COLOR_BGR2RGB)
            generation = wandb.Image(generation)


            detail_sketch_table.add_data(name, sketch, photo, generation, ssim_score, inception_score, fid_score)
        wandb.log({f"detail_sketches_epoch_{epoch+1}_{str(wandb.run.id)}": detail_sketch_table})
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
    

    def train(self, config=None, verbose=True):
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
        dataset_artifact.add_dir("model_input/mixed_/")
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
                    self.PSNR.update(output_train, targets)
                    self.SSIM.update(output_train, targets)
                    self.IS.update(output_train)
                    self.FID.update(output_train, real=False)
                    self.FID.update(targets, real=True)

                    # Compute gradients and update weights
                    loss_train.backward()
                    optimizer.step()

                    if verbose:
                        print(f'Epoch: {epoch+1}/{config.epochs} | Batch: {batch_idx+1}/{len(self.dataloader)} | Loss: {loss_train.item():.4f}')

                # Compute average loss for the epoch
                epoch_loss = epoch_loss / len(self.dataloader)
                psnr_train = self.PSNR.compute()
                ssim_train = self.SSIM.compute()
                is_train = self.IS.compute()
                fid_train = self.FID.compute()

                self.PSNR.reset()
                self.SSIM.reset()
                self.IS.reset()
                self.FID.reset()

                print(f'Epoch {epoch+1} Training Loss:   {epoch_loss}')

                # calculate validation loss and metrics
                
                with torch.no_grad():
                    output_valid = self.forward(test_X, test_condition)
                    loss_valid = self.loss(test_Y, output_valid)
                    print(f'Epoch {epoch+1} Validation Loss: {loss_valid}\n')

                    ssim_valid, fid_valid, is_valid, psnr_valid = self.calculate_metrics(output_valid, test_Y)

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
                    'valid-SSIM': ssim_valid,
                    'train-FID': fid_train,
                    'valid-FID': fid_valid,
                    'train-IS': is_train,
                    'valid-IS': is_valid
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

