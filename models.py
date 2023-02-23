"""
To run this template just do:
python gan.py
After a few epochs, launch TensorBoard to see the images being generated at every batch:
tensorboard --logdir default
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.utils.data import Dataset
from pytorch_lightning.core import LightningModule
from pytorch_lightning.core import LightningDataModule

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32,4)
        )

    def forward(self, matrix):
        return torch.reshape(self.fc_block(matrix), matrix.shape)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32,1) # outputs logit
        )

    def forward(self, matrix):
        return self.fc_block(matrix)

class GAN(LightningModule):

    def __init__(self, lr: float = 0.0002, intRange = int(1e6)):
        super().__init__()
        self.lr = lr
        self.intRange = intRange

        # networks
        self.generator = Generator()
        self.discriminator = Discriminator()

    def forward(self, matrix):
        return self.generator(matrix)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        #imgs, _ = batch
        matrices = batch
        # sample noise
        #z = torch.randn(imgs.shape[0], self.latent_dim)
        #z = z.type_as(imgs)
        z = torch.randint(
            low = -self.intRange,
            high = self.intRange,
            size = matrices.shape
            )
        z = z.type_as(matrices)

        # train generator
        if optimizer_idx == 0:

            # generate images
            #self.generated_imgs = self(z)
            self.generated_matrices = self(z)

            # log sampled images
            #sample_imgs = self.generated_imgs[:6]
            #grid = torchvision.utils.make_grid(sample_imgs)
            #self.logger.experiment.add_image('generated_images', grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            #valid = torch.ones(imgs.size(0), 1)
            #valid = valid.type_as(imgs)

            # generator training labels
            # all true (1) because we want generator to produce examples that the
            # discriminator classifies as true
            # so we should minimize the loss of being classified as true
            valid = torch.ones(matrices.size(0), 1)
            valid = valid.type_as(matrices)

            # adversarial loss is binary cross-entropy
            generator_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            return generator_loss

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(matrices.size(0), 1)
            valid = valid.type_as(matrices)

            real_loss = self.adversarial_loss(self.discriminator(matrices), valid)

            # how well can it label as fake?
            fake = torch.zeros(matrices.size(0), 1)
            fake = fake.type_as(matrices)

            #fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)
            fake_loss = self.adversarial_loss(self.discriminator(self(z)), fake)

            # discriminator loss is the average of these
            discriminator_loss = (real_loss + fake_loss) / 2
            return discriminator_loss

    def configure_optimizers(self):
        lr = self.lr

        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []
    
    def generate_samples(self, num_samples):
        with torch.no_grad():
            z = torch.randint(
                low = -self.intRange,
                high = self.intRange,
                size = (num_samples,2,2)
                ).to(torch.float32)
            return self(z)

class MatrixDataset(Dataset):
    def __init__(self, matrices):
        self.matrices = matrices

    def __len__(self):
        return self.matrices.shape[0]

    def __getitem__(self, idx):
        return self.matrices[idx]

class SL2Z_DataModule(LightningDataModule):
    def __init__(self, data_path, batch_size) -> None:
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size

    def setup(self, stage: str):
        if stage == 'fit':
            self.matrices = torch.tensor(np.load(self.data_path)).to(torch.float32)
    
    def train_dataloader(self):
        print('Creating train dataloader')
        return DataLoader(MatrixDataset(self.matrices), batch_size = self.batch_size)