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

from collections import OrderedDict
from pytorch_lightning.core import LightningModule
from pytorch_lightning.core import LightningDataModule
from torch.utils.data import Dataset

class Generator(nn.Module):
	def __init__(self, noise_dim):
		super().__init__()
		self.noise_dim = noise_dim
		latent_dim = noise_dim + noise_dim * (noise_dim + 1) // 2
		self.block = nn.Sequential(
			nn.Flatten(),
			nn.Linear(latent_dim, 4*latent_dim),
			nn.BatchNorm1d(4*latent_dim),
			nn.ReLU(),
			nn.Linear(4*latent_dim, 4)
		)

	def forward(self, noise):
		batch_size, noise_dim = noise.size()
		# create vector of degree 1 elements. size=(noise_dim,)
		deg1 = noise
		# create vector of degree 2 products. size=(noise_dim * (noise_dim + 1) // 2,)
		# Kronecker product applied to each entry in batch
		deg2 = torch.einsum(
			'ni,nj->nij',noise,noise
			).reshape(shape=(batch_size, noise_dim, noise_dim))
		# select upper triangular entries (only take unique products)
		mask = torch.triu(
			torch.ones(size=(noise_dim,noise_dim),dtype=bool)
			).to(device=noise.device)
		deg2 = torch.masked_select(deg2,mask).reshape(shape=(batch_size,-1))
		# concatenate
		out = torch.cat([deg1,deg2],dim=1)
		return torch.reshape(self.block(out), shape=(batch_size,2,2))

class Discriminator(nn.Module):
	def __init__(self, packing_degree):
		super().__init__()
		self.packing_degree = packing_degree
		self.mask = torch.triu(torch.ones(size=(4,4),dtype=bool))
		self.flatten = nn.Flatten()
		self.block = nn.Sequential(
			nn.Linear(self.packing_degree * 14, 14),
			nn.ReLU(),
			nn.Linear(14,1)
		)

	def forward(self, matrix):
		batch_size = matrix.size(dim=0)
		assert batch_size % self.packing_degree == 0
		# create vector of degree 1 elements. size=(4,)
		deg1 = self.flatten(matrix)
		# create vector of degree 2 products. size=(10,)
		# Kronecker product applied to each entry in batch
		deg2 = torch.einsum('ni,nj->nij',deg1,deg1)
		# select upper triangular entries (only take unique products)
		mask = torch.triu(
			torch.ones(size=(4,4),dtype=bool)
			).to(device=matrix.device)
		deg2 = torch.masked_select(deg2,mask).reshape(shape=(batch_size,-1))
		# concatenate
		out = torch.cat([deg1,deg2],dim=1)
		# implement packing: concatenate packing_degree number of features together
		out = out.view(batch_size // self.packing_degree,-1)
		# out.size() = (batch_size // self.packing_degree, 1)
		# need to output label for each matrix in split
		out = torch.vstack(tuple(self.block(out) for _ in range(self.packing_degree)))
		out = out.T.reshape((batch_size,1))

		return out

class IntegralityLoss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, matrices):
		out = matrices - torch.round(matrices)
		out = torch.linalg.norm(out, dim=(1,2))
		out = torch.mean(out)
		return out

class DeterminantLoss(nn.Module):
	def __init__(self):
		super().__init__()
	
	def forward(self, matrices):
		out = matrices[:,0,0] * matrices[:,1,1] - matrices[:,0,1] * matrices[:,1,0]
		out = out - 1
		out = torch.square(out)
		out = torch.sum(out)
		out = torch.sqrt(out)
		return out

class GAN(LightningModule):
	def __init__(self, config):
		super().__init__()
		self.lr = config['lr']
		self.noise_dim = config['noise_dim']
		self.noise_sigma = config['noise_sigma']
		self.packing_degree = config['packing_degree']
		# networks
		self.generator = Generator(self.noise_dim)
		self.discriminator = Discriminator(self.packing_degree)
		# loss
		self.adversarial_loss = nn.BCEWithLogitsLoss()
		self.integrality_loss = IntegralityLoss()
		self.determinant_loss = DeterminantLoss()

	def forward(self, matrix):
		return self.generator(matrix)
	
	def gradient_penalty(self, real_matrices, fake_matrices):
		"""Calculates the gradient penalty loss for WGAN GP"""
		# Random weight term for interpolation between real and fake samples
		alpha = torch.Tensor(np.random.random((real_matrices.size(0), 1, 1))).to(self.device)
		# Get random interpolation between real and fake samples
		interpolates = (alpha * real_matrices + ((1 - alpha) * fake_matrices)).requires_grad_(True)
		interpolates = interpolates.to(self.device)
		d_interpolates = self.discriminator(interpolates)
		fake = torch.Tensor(real_matrices.shape[0], 1).fill_(1.0).to(self.device)
		# Get gradient w.r.t. interpolates
		gradients = torch.autograd.grad(
			outputs=d_interpolates,
			inputs=interpolates,
			grad_outputs=fake,
			create_graph=True,
			retain_graph=True,
			only_inputs=True,
		)[0]
		gradients = gradients.view(gradients.size(0), -1).to(self.device)
		gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
		return gradient_penalty

	def training_step(self, batch, batch_idx, optimizer_idx):
		matrices = batch
		batch_size = matrices.size(dim=0)
		# sample noise
		z = torch.normal(
			mean = 0,
			std=self.noise_sigma,
			size=(batch_size, self.noise_dim)
			)
		z = z.type_as(matrices)
		# generated matrices
		gen_matrices = self(z)
		# ground truths
		valid = torch.ones(size=(batch_size,1),requires_grad=False).to(matrices.device)
		fake = torch.zeros(size=(batch_size,1),requires_grad=False).to(matrices.device)
		# discriminator outputs
		real_scores = self.discriminator(matrices)
		fake_scores = self.discriminator(gen_matrices)
		real_rel_scores = real_scores - torch.mean(fake_scores, 0, keepdim = True)
		fake_rel_scores = fake_scores - torch.mean(real_scores, 0, keepdim = True)

		# train generator
		if optimizer_idx == 0:
			lambda_il = 100
			lambda_dl = 100
			# loss: reward generator for creating samples that fool generator
			real_gen_loss = self.adversarial_loss(real_rel_scores,fake)
			fake_gen_loss = self.adversarial_loss(fake_rel_scores, valid)
			integrality_loss = self.integrality_loss(gen_matrices)
			determinant_loss = self.determinant_loss(gen_matrices)
			generator_loss = real_gen_loss + fake_gen_loss + lambda_il * integrality_loss + lambda_dl * determinant_loss

			self.log('real_gen_loss', real_gen_loss, sync_dist=True)
			self.log('fake_gen_loss', fake_gen_loss, sync_dist=True)
			self.log('integrality_loss', integrality_loss, sync_dist=True)
			self.log('determinant_loss', determinant_loss, sync_dist=True)
			self.log('generator_loss', generator_loss, sync_dist=True)

			return generator_loss

		# train discriminator
		if optimizer_idx == 1:
			lambda_gp = 100
			# loss: penalize discriminator for mis-classifying real/fake
			real_disc_loss = self.adversarial_loss(real_rel_scores, valid)
			fake_disc_loss = self.adversarial_loss(fake_rel_scores, fake)
			gradient_penalty = self.gradient_penalty(matrices, gen_matrices)

			discriminator_loss = real_disc_loss + fake_disc_loss + lambda_gp * gradient_penalty

			self.log('real_disc_loss', real_disc_loss, sync_dist=True)
			self.log('fake_disc_loss', fake_disc_loss, sync_dist=True)
			self.log('discriminator_loss', discriminator_loss, sync_dist=True)
			self.log('gradient_penalty', gradient_penalty)
			
			return discriminator_loss

	def configure_optimizers(self):
		lr = self.lr

		opt_g = torch.optim.AdamW(self.generator.parameters(), lr=lr)
		opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=lr)
		return (
			{'optimizer': opt_g},
			{'optimizer': opt_d}
		)

	def generate_samples(self, num_samples):
		with torch.no_grad():
			z = torch.normal(
				mean = 0, 
				std = self.noise_sigma,
				size=(num_samples, self.noise_dim)
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
	def __init__(self, config) -> None:
		super().__init__()
		self.data_path = config['data_path']
		self.batch_size = config['batch_size']
		self.num_workers = config['num_workers']

	def setup(self, stage: str):
		if stage == 'fit':
			self.matrices = torch.tensor(np.load(self.data_path)).to(torch.float32)

	def train_dataloader(self):
		return DataLoader(
			MatrixDataset(self.matrices),
			batch_size = self.batch_size,
			num_workers = self.num_workers
			)
