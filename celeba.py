import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, dataset_info):
        super(CelebADataset, self).__init__()

        num_img = dataset_info['num_img']

        condition_size = dataset_info['condition_size']


        self.dataset_info = dataset_info


        self.X = sorted(os.listdir(dataset_info['source']))[:num_img]



        train_len = int(len(self.X) * dataset_info['train_split'])
        
        self.train_X = self.X[0:train_len]
        self.test_X = self.X[train_len:len(self.X)]

        self.Y = sorted(os.listdir(dataset_info['target']))[:num_img]
        self.train_Y = self.Y[0:train_len]
        self.test_Y = self.Y[train_len:len(self.Y)]

        if dataset_info['condition'] is not None:
            self.condition_data = np.load(dataset_info['condition'])
        else:
            self.condition_data = np.ones((len(self.X), 1))
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
            'resize_bw': torchvision.transforms.Resize((16, 16)),
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sketch_path = os.path.join(self.dataset_info['source'], self.train_X[idx])
        sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)

        # sketch_path = os.path.join(self.dataset_info['target'], self.train_X[idx])
        # sketch = cv2.imread(sketch_path)
        # sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)


        original_path = os.path.join(self.dataset_info['target'], self.train_Y[idx])
        original = cv2.imread(original_path)

        assert sketch_path.split('/')[-1] == original_path.split('/')[-1], 'Sketch and original image must have same name'

        sketch = self.transforms['to_tensor'](sketch)

        original = self.transforms['to_tensor'](original)

        # Sketches are already normalized, normalizing the original images will make them black
        # original = self.transforms['normalize_original'](original)
        # sketch = self.transforms['normalize_sketch'](sketch)

        original = self.transforms['resize_original'](original)
        sketch = self.transforms['resize_sketch'](sketch)

        # Golden standard with pytorch dipshit
        # cv2.imwrite('original.png', original.permute(1, 2, 0).numpy() * 255)

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

        sample_indices = np.random.choice(len(self.test_X),sample_size,False)
        sample_X = np.array(self.test_X)[sample_indices]
        sample_Y = np.array(self.test_Y)[sample_indices]
        sample_condition_vec = np.array(self.test_condition_data)[sample_indices]

        for sketch_path, original_path in zip(sample_X, sample_Y):
            sketch_path = os.path.join(self.dataset_info['source'], sketch_path)
            sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)

            # sketch_path = os.path.join(self.dataset_info['target'], sketch_path)
            # sketch = cv2.imread(sketch_path)
            # sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)

            original_path = os.path.join(self.dataset_info['target'], original_path)
            original = cv2.imread(original_path)

            sketch = self.transforms['to_tensor'](sketch)
            original = self.transforms['to_tensor'](original)
            
            # Sketches are already normalized, normalizing the original images will make them black
            # original = self.transforms['normalize_original'](original)
            # sketch = self.transforms['normalize_sketch'](sketch)

            original = self.transforms['resize_original'](original)
            sketch = self.transforms['resize_sketch'](sketch)

            X.append(sketch)
            Y.append(original)
        
        return torch.stack(X), torch.stack(Y), torch.from_numpy(sample_condition_vec)
        

        
        # get random permutation of images


        