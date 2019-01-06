
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np


manualSeed = random.randint(1,10000)

random.seed(manualSeed)

torch.manual_seed(manualSeed)

dataroot = "dataFolder/CelebADataset"

workers = 4

batch_size = 128

image_size = 64

image_channels = 3

latent_vector_dim = 100

generator_feature_maps = 64

discriminator_feature_maps = 64

dataset = dset.ImageFolder(root = dataroot, transform = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]))

dataloader = torch.utils.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = workers)

device = torch.device("cpu")


def initialize_weights(model):
	class_name = model.__class__.__name__
	if class_name.find('Conv')!=-1:
		nn.init.normal_(model.weight.data, 0.0, 0.02)

	elif class_name.find('BatchNorm') != -1: 
		nn.init.normal_(model.weight.data, 0.0, 0.02)
		nn.init.constant_(model.bias.data, 0)


class Generator(nn.Module):
	def __init__(self):
		super().__init__()
		self.container = nn.Sequential(nn.ConvTranspose2d(latent_vector_dim, generator_feature_maps*8, 4,1,0, bias = False), 
			nn.BatchNorm2d(generator_feature_maps*8), 
			nn.ReLU(True),
			nn.ConvTranspose2d(generator_feature_maps*8, generator_feature_maps*4, 4, 2, 1, bias = False),
			nn.BatchNorm2d(generator_feature_maps*4),
			nn.ReLU(True),
			nn.ConvTranspose2d(generator_feature_maps*4, generator_feature_maps*2, 4,2,1, bias = False),
			nn.BatchNorm2d(generator_feature_maps*2),
			nn.ReLU(True),
			nn.ConvTranspose2d(generator_feature_maps*2, generator_feature_maps, 4,2,1, bias = False), 
			nn.BatchNorm2d(generator_feature_maps),
			nn.ReLU(True),
			nn.ConvTranspose2d(generator_feature_maps, image_channels)
			nn.Tanh())  # Deconvolution kernel dimensions, weight initialization schemes may differ from the values specified in the original paper


	def forward(self, input):
		return self.container(input)

generator = Generator().to(device) # Move generator to CPU

generator.apply(initialize_weights) # Initialize the generator's weights



class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()
		self.container = nn.Sequential(nn.Conv2d(image_channels, discriminator_feature_maps, 4,2,1, bias = False), 
			nn.LeakyReLU(0.2, inplace = True), 
			nn.Conv2d(discriminator_feature_maps, discriminator_feature_maps*2, 4,2,1, bias = False),
			nn.BatchNorm2d(discriminator_feature_maps*2), 
			nn.LeakyReLU(0.2, inplace = True), 
			nn.Conv2d(discriminator_feature_maps*2, discriminator_feature_maps*4, 4,2,1, bias = False), 
			nn.BatchNorm2d(discriminator_feature_maps*4), 
			nn.LeakyReLU(0.2, inplace = True), 
			nn.Conv2d(discriminator_feature_maps*4, discriminator_feature_maps*8, 4,2,1, bias = False)
			nn.BatchNorm2d(discriminator_feature_maps*8), 
			nn.LeakyReLU(0.2, inplace = True), 
			nn.Conv2d(discriminator_feature_maps*8, 1,4,1,0, bias = False), 
			nn.Sigmoid())

	def forward(self, input):
		return self.container(input)


discriminator = Discriminator().to(device)

discriminator.apply(initialize_weights) # Initialize weights to mean 0 and variance 0.04

criterion = nn.BCELoss() # l(n) = -[yn.log(xn) + (1-yn).log(1-xn)]   

latent_noise = torch.randn(64, latent_vector_dim, 1,1, device = device)

training_label = 1 # Labels originating from the training set, i.e. real

generated_label = 0 # Fake labels

discriminator_optimizer = optim.SGD(discriminator.parameters(), lr = 0.01) # Try SGD in the discriminator (even though this flies in the face of the paper's suggestions)

generator_optimizer = optim.Adam(generator.parameters(), lr = 0.0002, betas = (0.5, 0.999)) # Adam in the generator

# Training loop

epochs = 20

generator_losses = []

discriminator_losses = []

for epoch in range(epochs):
	for index, data_batch in enumerate(dataloader, 0):
		discriminator.zero_grad()

		cpu_batch = data_batch[0].to(device)

		data_batch_size = cpu_batch.size(0)

		discriminator_labels = torch.full((data_batch_size,), training_label, device = device)

		discriminator_output = discriminator(cpu_batch).view(-1)

		real_batch_error = criterion(discriminator_output, discriminator_labels)

		real_batch_error.backward()

		latent_vector = torch.randn(data_batch_size, latent_vector_dim, 1,1, device = device)

		generator_fake_batch_outputs = generator(latent_vector)

		label.fill_(generated_label)

		discriminator_fake_predictions = discriminator(generator_fake_batch_outputs.detach()).view(-1) # Detach from the computational graph to avoid computing gradients w.r.t. the generator's parameters

		discriminator_error = criterion(discriminator_fake_predictions, label)
 
		discriminator_error.backward() # Accumulate gradients

		fake_prediction_mean = discriminator_fake_predictions.mean().item()

		aggregated_discriminator_error = real_batch_error + discriminator_error

		discriminator_optimizer.step()

		# Perform generator updates

		generator.zero_grad()

		label.fill_(training_label)

		discriminator_outputs = discriminator(generator_fake_batch_outputs).view(-1)

		generator_error = criterion(discriminator_outputs, label)

		generator_error.backward()

		generator_optimizer.step()

		generator_losses.append(generator_error.item())

		discriminator_losses.append(discriminator_error.item())



















		









 






