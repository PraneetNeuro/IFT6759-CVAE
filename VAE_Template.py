import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_size, output_size, condition_size, dataset_info={
        'source': 'source.npy',
        'target': 'target.npy',
        'condition': 'condition.npy',
        'train_split': 0.8,
    }, model_config={
        'condition': True,
        'condition_type': 'mul',
        'skip_connection': True,
        'distribution_dim': 128,
    }):
        super(VariationalAutoEncoder, self).__init__()

        self.distribution = torch.distributions.Normal(torch.zeros(model_config['distribution_dim']), torch.ones(model_config['distribution_dim']))

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        self.to(self.device)

        self.input_size = input_size
        self.output_size = output_size
        self.projection_size = input_size[0] * input_size[1]
        self.condition_size = condition_size # size of the condition vector and the bottleneck layer
        self.model_config = model_config

        self.X = np.load(dataset_info['source']) / 255.0
        self.X = torch.from_numpy(self.X).float().view(-1, 1, self.input_size[0], self.input_size[1]).to(self.device)

        self.Y = np.load(dataset_info['target']) / 255.0
        self.Y = torch.from_numpy(self.Y).float().view(-1, 1, self.output_size[0], self.output_size[1]).to(self.device)

        if dataset_info['condition'] is not None:
            self.condition_data = np.load(dataset_info['condition'])
        else:
            if self.model_config['condition_type'] == 'add':
                self.condition_data = np.zeros((len(self.X), self.condition_size))
            else:
                self.condition_data = np.ones((len(self.X), self.condition_size))
        self.condition_data = torch.from_numpy(self.condition_data).float().view(-1, self.condition_size).to(self.device)

        self.train_X = self.X[:int(len(self.X) * dataset_info['train_split'])]
        self.train_Y = self.Y[:int(len(self.Y) * dataset_info['train_split'])]
        self.train_condition = self.condition_data[:int(len(self.condition_data) * dataset_info['train_split'])]

        self.test_X = self.X[int(len(self.X) * dataset_info['train_split']):]
        self.test_Y = self.Y[int(len(self.Y) * dataset_info['train_split']):]
        self.test_condition = self.condition_data[int(len(self.condition_data) * dataset_info['train_split']):]


        # Encoder layers

        # bottleneck and projection layers
        self.bottle_neck = nn.LazyLinear(condition_size)
        self.mean = torch.nn.Linear(condition_size, self.model_config['distribution_dim'])
        self.log_std = torch.nn.Linear(condition_size, self.model_config['distribution_dim'])

        # Decoder layers
        self.projection_layer = nn.Linear(self.model_config['distribution_dim'], self.projection_size)

    def encoder(self, x, condition):
        # Encoder Forward Pass
        
        # high dimensional projection through linear layer
        x = torch.flatten(x, 1)
        x = self.bottle_neck(x)
        if not self.model_config['condition']:
            condition = torch.ones(x.size())
        if self.model_config['condition_type'] == 'add':
            x = torch.add(x, condition)
        else:
            x = torch.mul(x, condition)
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = torch.exp(0.5 * log_std)
        eps = torch.randn_like(std)
        z = mean + std * eps
        return z, mean, log_std
    
    def decoder(self, encoded_representation, input):
        x = self.projection_layer(x)
        x = encoded_representation.view(-1, self.input_size[0], self.input_size[1])
        if self.model_config['skip_connection']:
            x = torch.cat((x, input), dim=1)
        else:
            x = x
        # Decoder Forward Pass, make sure the number of channels / kernels is 3, since  the output is a RGB image

        # Use the sigmoid function for the final layer
        pass
    
    def forward(self, input, condition):
        z, mean, log_std = self.encoder(input, condition)
        x = self.decoder(x, input)
        return z, mean, log_std, x
    
    def loss(self, generated_image, ground_truth, mean, log_std):
        reconstruction_loss = F.mse_loss(generated_image, ground_truth)
        latent_distrbution = torch.distributions.Normal(mean, log_std.exp())
        kl_divergence = F.kl_div(latent_distrbution, self.distribution)

        return reconstruction_loss + kl_divergence

    def train(self, epochs, batch_size, save_path):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(epochs):
            for i in range(0, len(self.train_X), batch_size):
                batch_X = self.train_X[i:i+batch_size]
                batch_Y = self.train_Y[i:i+batch_size]
                batch_condition = self.train_condition[i:i+batch_size]
                optimizer.zero_grad()
                _, mean, log_std, output = self.forward(batch_X, batch_condition)
                loss = self.loss(output, batch_Y, mean, log_std)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch} Loss: {loss}')
            with torch.no_grad():
                _, mean, log_std, output = self.forward(self.test_X, self.test_condition)
                loss = self.loss(self.test_Y, output, mean, log_std)
                print(f'Validation Loss: {loss}')
            self.save_model(save_path)

    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
