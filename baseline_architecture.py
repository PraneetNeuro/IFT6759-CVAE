import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import DataLoader

class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, dataset_info):
        super(CelebADataset, self).__init__()

        input_size = dataset_info['input_size']
        output_size = dataset_info['output_size']
        condition_size = dataset_info['condition_size']

        self.X = np.load(dataset_info['source'])
        self.X = self.X - np.mean(self.X) / np.std(self.X)
        self.X = torch.from_numpy(self.X).float().view(-1, 1, input_size[0], input_size[1])
        train_len = int(len(self.X) * dataset_info['train_split'])
        
        self.train_X = self.X[0:train_len]
        self.test_X = self.X[train_len:len(self.X)]

        self.Y = np.load(dataset_info['target'])
        self.Y = self.Y - np.mean(self.Y) / np.std(self.Y)
        self.Y = torch.from_numpy(self.Y).float().view(-1, 3, output_size[0], output_size[1])
        self.train_Y = self.Y[0:train_len]
        self.test_Y = self.Y[train_len:len(self.Y)]

        if dataset_info['condition'] is not None:
            self.condition_data = np.load(dataset_info['condition'])
        else:
            self.condition_data = np.ones((len(self.X), condition_size))
        self.condition_data = torch.from_numpy(self.condition_data).float().view(-1, condition_size)

        self.train_condition_data = self.condition_data[0:train_len]
        self.test_condition_data = self.condition_data[train_len:len(self.condition_data)]

        assert self.X.shape[0] == self.Y.shape[0] == self.condition_data.shape[0], 'Number of samples in X, Y and condition data must be equal'

        self.num_samples = self.train_condition_data.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.train_X[idx], self.train_Y[idx], self.train_condition_data[idx]
    
    def get_test_data(self):
        return self.test_X, self.test_Y, self.test_condition_data

class AutoEncoder(nn.Module):
    def __init__(self, input_size, output_size, condition_size=1024, dataset_info={
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
            self.device = torch.device('mps')
            # self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        self.input_size = input_size
        self.output_size = output_size
        self.projection_size = input_size[0] * input_size[1]
        self.condition_size = condition_size # size of the condition vector and the bottleneck layer
        self.model_config = model_config

        self.dataset = CelebADataset({
            'source': dataset_info['source'],
            'target': dataset_info['target'],
            'condition': dataset_info['condition'],
            'train_split': dataset_info['train_split'],
            'input_size': input_size,
            'output_size': output_size,
            'condition_size': condition_size,
        })

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

        self.to(self.device)

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
        if self.model_config['condition_type'] == 'add':
            x = torch.add(x, condition)
        else:
            print(x.shape, condition.shape)
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

    def train(self, epochs, batch_size, save_path):
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        test_X, test_Y, test_condition = self.dataset.get_test_data()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(epochs):
            for batch_idx, (images, conditions, targets) in enumerate(self.dataloader):
                images, targets, conditions = images.to(self.device), conditions.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                output = self.forward(images, conditions)
                loss = self.loss(targets, output)
                loss.backward()
                optimizer.step()
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}")
                    
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            with torch.no_grad():
                output = self.forward(test_X.to(self.device), test_condition.to(self.device))
                loss = self.loss(test_Y.to(self.device), output)
                print(f"Validation Loss: {loss:.4f}")
            self.save_model(save_path)

    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))

ae = AutoEncoder(input_size=(128, 128), output_size=(268, 268), condition_size=1024)
ae.train(epochs=10, batch_size=350, save_path='model.pth')
