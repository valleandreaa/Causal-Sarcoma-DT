import os
import numpy as np
import torch
import torch.nn as nn

from models.weight_init import initialize_weights_kaiming, initialize_weights_xavier, initialize_weights_normal
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class SimpleGANGenerator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleGANGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 96),  # Slightly larger intermediate layer
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Linear(96, 128),  # Gradual expansion
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.Linear(160, 128),  # Start reducing dimensions
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid() 
        )

    def forward(self, x):
        mask = ~torch.isnan(x)
        
        # Replace NaN values with 0 for safe computation in the model
        x = torch.where(mask, x, torch.zeros_like(x))
        
        # Ensure x is on the same device as the model's parameters
        # device = next(self.model.parameters()).device
        # x = x.to(device)
        
        return self.model(x)

class LSTMGenerator(nn.Module):
    """
    LSTM-based Generator that takes a main input sequence and a conditioning sequence
    and produces an output sequence of the same time length.
    Uses attention-based conditioning for more effective feature interaction.
    """
    def __init__(self, input_dim, cond_input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMGenerator, self).__init__()
        # LSTM processes concatenated input
        combined_input_dim = (input_dim + cond_input_dim)*2
        self.lstm = nn.LSTM(input_size=combined_input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True)
        
        

        # Linear layer maps conditioned hidden states to output dimension
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, m_x, cond, m_cond, lengths=None, hidden=None):
       # 1) zero-out should already be done: x = x * m_x; cond = cond * m_cond

        # 2) augment each vector with its mask flags
        x_aug    = torch.cat([x,    m_x],    dim=-1)  # → (B, T, 2·D_x)
        cond_aug = torch.cat([cond, m_cond], dim=-1)  # → (B, T, 2·D_c)

        # 3) full input into LSTM = [x_aug ⊕ cond_aug]
        combined = torch.cat([x_aug, cond_aug], dim=-1)  # (B, T, total_dim)

        # 4) pack to skip fully-padded timesteps
        if lengths is not None:
            packed, _    = pack_padded_sequence(combined, lengths.cpu(),
                                                 batch_first=True,
                                                 enforce_sorted=False)
            packed_out, h = self.lstm(packed, hidden)
            lstm_out, _   = pad_packed_sequence(packed_out,
                                                batch_first=True,
                                                total_length=combined.size(1))
        else:
            lstm_out, h = self.lstm(combined, hidden)

        # 5) project per timestep
        out = self.fc(lstm_out)  # (B, T, output_dim)
        return out, h

class LSTMDiscriminator(nn.Module):
    """
    LSTM-based Discriminator that evaluates a sequence and outputs a single score.
    """
    def __init__(self, input_dim, cond_dim, hidden_dim, output_dim,  num_layers=1):
        super(LSTMDiscriminator, self).__init__()
        # LSTM to process the input sequence
        total_dim = (input_dim + cond_dim) * 2

        self.lstm = nn.LSTM(
            input_size=total_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        # Linear layer for final binary classification (or regression) output
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self,
                x:       torch.Tensor,
                m_x:     torch.Tensor,
                cond:    torch.Tensor,
                m_cond:  torch.Tensor,
                lengths: torch.Tensor  = None):
        # 1) augment each timestep with its mask
        x_aug    = torch.cat([x,    m_x],    dim=-1)  # (B,T,2D_x)
        cond_aug = torch.cat([cond, m_cond], dim=-1)  # (B,T,2D_c)
        combined = torch.cat([x_aug, cond_aug], dim=-1)

        # 2) pack padded timesteps if needed
        if lengths is not None:
            packed, _         = pack_padded_sequence(
                                    combined, lengths.cpu(),
                                    batch_first=True,
                                    enforce_sorted=False)
            packed_out, (h_n, _) = self.lstm(packed)
            # restore padded shape (we only need h_n though)
            _, _             = pad_packed_sequence(
                                    packed_out,
                                    batch_first=True,
                                    total_length=combined.size(1))
        else:
            _, (h_n, _)      = self.lstm(combined)

        # 3) h_n: (num_layers, B, hidden_dim) → take last layer
        last_hidden = h_n[-1]                        # (B, hidden_dim)

        # 4) classify
        score = self.classifier(last_hidden).squeeze(-1)  # (B,)
        return score


class SimpleGANDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(SimpleGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
        
            nn.Linear(input_dim, 64),
            # nn.LayerNorm(64),  # Normalization for stability
            nn.LeakyReLU(0.2),
            
            nn.Linear(64, 1),
            nn.Sigmoid()  # Binary classification for real/fake
        )

    def forward(self, x):
        mask = ~torch.isnan(x)
        
        # Replace NaN values with 0 for safe computation in the model
        x = torch.where(mask, x, torch.zeros_like(x))
        
        # Ensure x is on the same device as the model's parameters
        device = next(self.model.parameters()).device
        x = x.to(device)
        
        return self.model(x)


class TreatmentGANGenerator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TreatmentGANGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),  # Batch normalization
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),  # Batch normalization
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),  # Batch normalization
            nn.ReLU(inplace=True),
            # nn.Linear(256, 512),
            # nn.BatchNorm1d(512),  # Batch normalization
            # nn.ReLU(inplace=True),
            # nn.Linear(512, 256),
            # nn.BatchNorm1d(256),  # Batch normalization
            # nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # Batch normalization
            nn.ReLU(inplace=True),
            nn.Linear(128, 64), 
            nn.LayerNorm(64),  # Batch normalization
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
            nn.Tanh()  # Use Tanh for smoother outputs instead of Softmax
        )

    def forward(self, x):
        return self.model(x)


class TreatmentGANDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(TreatmentGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class CycleGANGenerator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CycleGANGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim)  # Output dimension for the target domain
        )

    def forward(self, x):
        return self.model(x)


# CycleGAN Discriminator for both domain A and domain B
class CycleGANDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(CycleGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),  # Binary output (real or fake)
            nn.Sigmoid()  # Sigmoid for output between 0 and 1
        )

    def forward(self, x):
        return self.model(x)

class GANGenerator(nn.Module):
    def __init__(self, input_dim, num_features, input_size, debug=False):
        super(GANGenerator, self).__init__()
        self.debug = debug
        self.input_size = input_size  # Set input size to 1 for the moment
        self.linear = nn.Sequential(nn.Linear(input_dim, num_features))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(num_features, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.input_size, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        if self.debug:
            print("GANGenerator layers:")
        x = self.linear(x)
        if self.debug:
            print(self.linear)
        x = x.view(x.shape[0], -1, self.input_size, self.input_size)
        out = self.conv_blocks(x)
        if self.debug:
            for layer in self.conv_blocks:
                print(layer)
        return out

# Discriminator model definition
class GANDiscriminator(nn.Module):
    def __init__(self, debug=False):
        super(GANDiscriminator, self).__init__()
        self.debug = debug
        def disc_module(input_channels, output_channels, batch_norm=True):
            layers = [nn.Conv2d(input_channels, output_channels, 3, 2, 1),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Dropout2d(0.25)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(output_channels, 0.8))
            return layers

        self.discriminator_model = nn.Sequential(
            *disc_module( 1, 16, batch_norm=False),
            *disc_module(16, 32),
            *disc_module(32, 64),
            *disc_module(64, 128)
        )
        ds_size = 1// 2 ** 4
        self.adversarial_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, x):
        if self.debug:
            print("GANDiscriminator layers:")
        x = self.discriminator_model(x)
        if self.debug:
            for layer in self.discriminator_model:
                print(layer)
        x = x.view(x.shape[0], -1)
        out = self.adversarial_layer(x)
        if self.debug:
            print(self.adversarial_layer)
        return out

    def _initialize_weights(self):
        """
        Initialize the weights of the model.
        """
        if self.init_method == 'kaiming':
            self.apply(initialize_weights_kaiming)
        elif self.init_method == 'xavier':
            self.apply(initialize_weights_xavier)
        elif self.init_method == 'normal':
            self.apply(initialize_weights_normal)
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")


class WGANGenerator(nn.Module):
    def __init__(self, latent_dim, img_shape, debug=False):
        super(WGANGenerator, self).__init__()
        self.debug = debug
        self.img_shape = img_shape

        def generator_block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *generator_block(latent_dim, 128, normalize=False),
            *generator_block(128, 256),
            *generator_block(256, 512),
            *generator_block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        if self.debug:
            print("WGANGenerator layers:")
        img = self.model(z)
        if self.debug:
            for layer in self.model:
                print(layer)
        img = img.view(img.size(0), *self.img_shape)
        return img

class WGANDiscriminator(nn.Module):
    def __init__(self, img_shape, debug=False):
        super(WGANDiscriminator, self).__init__()
        self.debug = debug
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        if self.debug:
            print("WGANDiscriminator layers:")
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        if self.debug:
            for layer in self.model:
                print(layer)
        return validity
    
    def _initialize_weights(self):
        """
        Initialize the weights of the model.
        """
        if self.init_method == 'kaiming':
            self.apply(initialize_weights_kaiming)
        elif self.init_method == 'xavier':
            self.apply(initialize_weights_xavier)
        elif self.init_method == 'normal':
            self.apply(initialize_weights_normal)
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")


class ResnetBlock(nn.Module):
    def __init__(self, dim, debug=False):
        super(ResnetBlock, self).__init__()
        self.debug = debug
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        if self.debug:
            print("ResnetBlock layers:")
            for layer in self.conv_block:
                print(layer)
        return x + self.conv_block(x)

# class CycleGANGenerator(nn.Module):
#     def __init__(self, input_nc, output_nc, n_residual_blocks=9, debug=False):
#         super(CycleGANGenerator, self).__init__()
#         self.debug = debug
#         model = [
#             nn.Conv2d(input_nc, 64, kernel_size=7, padding=3),
#             nn.InstanceNorm2d(64),
#             nn.ReLU(inplace=True)
#         ]
#         # Downsampling
#         in_features = 64
#         out_features = in_features * 2
#         for _ in range(2):
#             model += [
#                 nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
#                 nn.InstanceNorm2d(out_features),
#                 nn.ReLU(inplace=True)
#             ]
#             in_features = out_features
#             out_features = in_features * 2
#         # Residual blocks
#         for _ in range(n_residual_blocks):
#             model += [ResnetBlock(in_features, debug)]
#         # Upsampling
#         out_features = in_features // 2
#         for _ in range(2):
#             model += [
#                 nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
#                 nn.InstanceNorm2d(out_features),
#                 nn.ReLU(inplace=True)
#             ]
#             in_features = out_features
#             out_features = in_features // 2
#         model += [nn.Conv2d(64, output_nc, kernel_size=7, padding=3), nn.Tanh()]
#         self.model = nn.Sequential(*model)

#     def forward(self, x):
#         if self.debug:
#             print("CycleGANGenerator layers:")
#             for layer in self.model:
#                 x = layer(x)
#                 print(layer)
#             return x
#         else:
#             return self.model(x)
        
#     def _initialize_weights(self):
#         """
#         Initialize the weights of the model.
#         """
#         if self.init_method == 'kaiming':
#             self.apply(initialize_weights_kaiming)
#         elif self.init_method == 'xavier':
#             self.apply(initialize_weights_xavier)
#         elif self.init_method == 'normal':
#             self.apply(initialize_weights_normal)
#         else:
#             raise ValueError(f"Unknown initialization method: {self.init_method}")


# class CycleGANDiscriminator(nn.Module):
#     def __init__(self, input_nc, debug=False):
#         super(CycleGANDiscriminator, self).__init__()
#         self.debug = debug
#         self.model = nn.Sequential(
#             nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
#             nn.InstanceNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
#             nn.InstanceNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
#         )
#         self._initialize_weights()

#     def forward(self, x):
#         if self.debug:
#             print("CycleGANDiscriminator layers:")
#             for layer in self.model:
#                 x = layer(x)
#                 print(layer)
#             return x
#         else:
#             return self.model(x)

#     def _initialize_weights(self):
#         """
#         Initialize the weights of the model.
#         """
#         if self.init_method == 'kaiming':
#             self.apply(initialize_weights_kaiming)
#         elif self.init_method == 'xavier':
#             self.apply(initialize_weights_xavier)
#         elif self.init_method == 'normal':
#             self.apply(initialize_weights_normal)
#         else:
#             raise ValueError(f"Unknown initialization method: {self.init_method}")


# Cycle Wasserstein Regression GAN (CWR-GAN) models
# class CWRGANGenerator(nn.Module):
#     def __init__(self):
#         super(CWRGANGenerator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(LATENT_DIM, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, IMAGE_SIZE * IMAGE_SIZE * CHANNELS),
#             nn.Tanh()
#         )

#     def forward(self, z):
#         img = self.model(z)
#         img = img.view(img.size(0), CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
#         return img

# class CWRGANDiscriminator(nn.Module):
#     def __init__(self):
#         super(CWRGANDiscriminator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(IMAGE_SIZE * IMAGE_SIZE * CHANNELS, 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, 128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(128, 1)
#         )

#     def forward(self, img):
#         img_flat = img.view(img.size(0), -1)
#         validity = self.model(img_flat)
#         return validity

# class CWRGANRegressor(nn.Module):
#     def __init__(self):
#         super(CWRGANRegressor, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(IMAGE_SIZE * IMAGE_SIZE * CHANNELS, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 1)
#         )

#     def forward(self, img):
#         img_flat = img.view(img.size(0), -1)
#         regression = self.model(img_flat)
#         return regression