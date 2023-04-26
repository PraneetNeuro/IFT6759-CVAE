import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, config):
        super(CelebADataset, self).__init__()

        # get parameter dicts for the dataset from config
        self.data_source_config = config['data_source_config']
        self.data_config = config['data_config']
        # self.training_config = config['training_config']
        self.wandb_config = config['wandb_config']
        self.image_gen_config = config['image_gen_config']

        # get parameter values
        self.num_img = self.data_config['num_img']
        self.condition_size = self.data_config['condition_size']
        self.original_path = self.data_source_config['input_original_path']
        self.sketch_path = self.data_source_config['input_sketch_path']
        # self.condition_path = self.data_source_config['condition_path']
        self.input_size = self.data_config['input_size'][0]
        self.output_size = self.data_config['output_size'][0]

        # Load and split images for training and testing
        self.X = sorted(os.listdir(self.sketch_path))[:self.num_img]
        train_len = int(len(self.X) * self.data_config['train_split'])
        self.train_X = self.X[0:train_len]
        self.test_X = self.X[train_len:len(self.X)]
        self.Y = sorted(os.listdir(self.original_path))[:self.num_img]
        self.train_Y = self.Y[0:train_len]
        self.test_Y = self.Y[train_len:len(self.Y)]

        # Load the conditions data
        # if self.condition_path is not None:
        #     self.condition_data = np.load(self.condition_path)
        # else:
        #     self.condition_data = np.ones((len(self.X), 1))

        # process and split conditioning data
        # self.condition_data = torch.from_numpy(self.condition_data).float().view(-1, self.condition_size)[:self.num_img]
        # self.train_condition_data = self.condition_data[0:train_len]
        # self.test_condition_data = self.condition_data[train_len:len(self.condition_data)]

        # Check if the number of samples in sketches, original images and condition data is equal
        assert len(self.X) == len(self.Y), 'Number of samples in X, and Y must be equal'

        self.num_samples = train_len

        # Define the image preprocessing functions
        self.transforms = {
            'to_tensor': torchvision.transforms.ToTensor(),
            'normalize_original': torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            'normalize_sketch': torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
            'resize_original': torchvision.transforms.Resize(self.output_size, antialias=True),
            'resize_sketch': torchvision.transforms.Resize(self.input_size, antialias=True),
            'resize_bw': torchvision.transforms.Resize((16, 16), antialias=True),
        }



    def __len__(self):
        """
        Returns the number of samples in the dataset
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
        Gets the item at the given index `idx` in the dataset. This function is called by the PyTorch dataloader to fetch
        the data samples to be used in training, validation or testing. 
        
        Args:
        - idx (int): The index of the data sample to fetch from the dataset.
        
        Returns:
        - A tuple containing the following three elements:
            - sketch (torch.Tensor): The sketch image as a PyTorch tensor
            - original (torch.Tensor): The original image as a PyTorch tensor. 
            - condition_data (torch.Tensor): The condition data for the sample as a PyTorch tensor. 
        """
        # Load the sketch image
        sketch_path = os.path.join(self.sketch_path, self.train_X[idx])
        sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)

        # Load the original image
        original_path = os.path.join(self.original_path, self.train_Y[idx])
        original = cv2.imread(original_path)

        # Assert that the sketch and original images have the same filename
        assert sketch_path.split('/')[-1] == original_path.split('/')[-1], 'Sketch and original image must have same name'

        # Convert the images to PyTorch tensors
        sketch = self.transforms['to_tensor'](sketch)
        original = self.transforms['to_tensor'](original)

        # Resize the images
        original = self.transforms['resize_original'](original)
        sketch = self.transforms['resize_sketch'](sketch)

        # Get the condition data for this sample
        # condition_data = self.train_condition_data[idx]
        condition_data = torch.ones((1, 1))

        return sketch, original, condition_data
    


    def get_validation_sample(self, sample_size):
        """
        Generate a validation sample of specified size by randomly selecting sketches and their corresponding original
        images from the test set. Returns the sketches and original images in tensor format, along with their 
        corresponding condition data.

        Args:
            sample_size (int): size of the validation sample to generate

        Returns:
            torch.Tensor: tensor of sketches
            torch.Tensor: tensor of original images
            torch.Tensor: tensor of condition data
        """
        X = []
        Y = []

        # Select random indices from the test set without replacement
        sample_indices = np.random.choice(len(self.test_X),sample_size,False)

        # Get the filenames corresponding to the selected indices
        sample_X = np.array(self.test_X)[sample_indices]
        sample_Y = np.array(self.test_Y)[sample_indices]
        # sample_condition_vec = np.array(self.test_condition_data)[sample_indices]
        sample_condition_vec = np.ones((sample_size, 1))

        for sketch_file, original_file in zip(sample_X, sample_Y):

            # read sketches
            sketch_path = os.path.join(self.sketch_path, sketch_file)
            sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)

            # read target photos
            original_path = os.path.join(self.original_path, original_file)
            original = cv2.imread(original_path)

            # transform images
            sketch = self.transforms['to_tensor'](sketch)
            original = self.transforms['to_tensor'](original)
            
            # Resize the images
            original = self.transforms['resize_original'](original)
            sketch = self.transforms['resize_sketch'](sketch)

            X.append(sketch)
            Y.append(original)

        # Convert lists to tensors
        X = torch.stack(X)
        Y = torch.stack(Y)
        sample_condition_vec = torch.from_numpy(sample_condition_vec)

        return X, Y, sample_condition_vec
        
