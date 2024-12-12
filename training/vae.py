# container script for architecture and loss development for a variational autoencoder
from torch import nn, randn_like, sum
import numpy as np
import math

class KullbackLeiblerReconstructionLoss(nn.Module):
    def __init__(self, recon_loss: nn.Module, n_epochs: int, b_cycle_time: int = 3, b_peak_time: int = 2):
        super().__init__()
        self.recon_loss_func = recon_loss

        self._b = None
        self._b_schedule = None
        self._get_b_schedule(n_epochs, b_cycle_time, b_peak_time)
        
        
    def _get_b_schedule(self, n_epochs: int, b_cycle_time: int, b_peak_time: int):
        cycle = np.linspace(0.0, 1.0, num=b_cycle_time).tolist() + ([1.0]*(b_peak_time-1))
        n_cycles = math.ceil(n_epochs / len(cycle))

        self._b_schedule = cycle*n_cycles
        print(f"using beta schedule: {self._b_schedule}")

    def step(self, epoch):
        self._b = self._b_schedule[epoch]

    def kullback_leibler_loss(self, mean, logvar):
        # based on the implementation shown here:
        # https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f
        return -0.5 * sum(1 + logvar - mean.pow(2) - logvar.exp())

    def forward(self, x_hat, x, mean, logvar):
        r_loss = self.recon_loss_func(x_hat, x) # input, target
        
        # applying log transform to kl_loss to reduce likelihood of overflow
        # kl_loss = math.log(self.kullback_leibler_loss(mean, logvar)+1)
        kl_loss = math.log(abs(self.kullback_leibler_loss(mean, logvar)))
        # report both weighted/unweighted loss so model saving can always be equally weighted
        return r_loss+(self._b*kl_loss), r_loss+kl_loss
    

class VariationalAutoEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, latent_dim: int = 768):
        super().__init__()
        self.encoder = VariationalEncoder(encoder, latent_dim)
        self.decoder = AutoDecoder()

    def forward(self, x):
        z, mean, logvar = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, mean, logvar


class VariationalEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, latent_dim: int = 768):
        super().__init__()
        self.encoder = encoder
        self.conv_1 = nn.Conv2d(self.encoder.encoder._out_channels, latent_dim, kernel_size=3, stride=1, padding=0) # out: (b, 768, 2, 2)
        self.flatten = nn.Flatten()

        # embedding dense layers
        self.embed_in = nn.Linear(latent_dim*2*2, latent_dim*2)# assumes 2*2 featuremap at conv_1 output
        self.act_1 = nn.ReLU() # recommended over standard relu for vae?
        self.embed_out = nn.Linear(latent_dim*2, latent_dim) # extract at this layer for embeddings?
        self.act_2 = nn.ReLU()

        # mean/logvar dense layers
        self.mean_linear = nn.Linear(latent_dim, latent_dim)
        self.logvar_linear = nn.Linear(latent_dim, latent_dim)

    def encoder_forward(self, x):
        x = self.encoder(x)
        x = self.conv_1(x)
        x = self.flatten(x)

        # dense layers at embedding dim
        x = self.act_1(self.embed_in(x))
        x = self.act_2(self.embed_out(x))

        # mean/logvar dense layers
        mean = self.mean_linear(x)
        logvar = self.logvar_linear(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = randn_like(logvar)
        z = mean + logvar*eps
        return z

    def forward(self, x):
        mean, logvar = self.encoder_forward(x)
        z = self.reparameterize(mean, logvar)
        return z, mean, logvar
        


class AutoDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # input is (b, 768)
        self.conv_0 = nn.ConvTranspose2d(768, 512, kernel_size=4, stride=4, padding=0)
        self.relu_0 = nn.LeakyReLU(0.2)
                 
        # expects (b, 512, 4, 4) to produce (b, 3, 128, 128)
        self.conv_1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.relu_1 = nn.LeakyReLU(0.2)
        
        self.conv_2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.relu_2 = nn.LeakyReLU(0.2)
        
        self.conv_3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.relu_3 = nn.LeakyReLU(0.2)
        
        self.conv_4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.relu_4 = nn.LeakyReLU(0.2)
        
        self.conv_5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        self.sigmoid_1 = nn.Sigmoid()  # map output to [0, 1] interval

    def forward(self, x):
        # input: (b, 768)
        x = x.view(*x.shape, 1, 1) # (b, 768, 1, 1)
        x = self.conv_0(x) # (b, 512, 4, 4)
        x = self.relu_0(x)
        
        x = self.conv_1(x)
        x = self.relu_1(x)

        x = self.conv_2(x)
        x = self.relu_2(x)

        x = self.conv_3(x)
        x = self.relu_3(x)

        x = self.conv_4(x)
        x = self.relu_4(x)

        x = self.conv_5(x)
        x = self.sigmoid_1(x)
        return x
