import torch

import torch.nn as nn

import torch.nn.Functional as F

class Network(nn.module):

	def__init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1,6,5)
		self.conv2 = nn.Conv2d(6,16,5)
		# Affine transformations
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120,84)
		self.fc3 = nn.Linear(84,10)

	def forward(self,x):
		x = F.max_pool2d(F.relu(self.conv1(x), (2,2)))
		x = F.max_pool2d(F.relu(self.conv2(x)),2)
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self,x):
		dimensions = x.size()[1:] # Ignore the batch dimension
		num_features = 1
		for dimension in dimensions:
			num_features*=dimension

		return num_features



network = Network()

input = torch.randn(1,1,32,32)

outputs = network(input)

targets = torch.randn(10) # Randomly generated target tensor

targets = targets.view(1,-1)

loss_fn = nn.MSELoss()

loss = loss_fn(outputs, targets)

network.zero_grad() # Flush existing parametric gradient buffers

loss.backward() # Backpropagate the error

learning_rate = 0.01

for parameter in network.parameters(): # Custom optimization loop(in lieu of torch.optim)
	parameter.data.sub_(parameter.grad.delta*learning_rate)









