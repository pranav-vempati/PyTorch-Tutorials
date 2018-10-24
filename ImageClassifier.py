import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
training_set = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
trainloader = torch.utils.data.DataLoader(training_set, batch_size = 4, shuffle = True, num_workers = 2)
testing_set = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)
testloader = torch.utils.data.DataLoader(testing_set, batch_size = 4, shuffle = False, num_workers = 2) 
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck') # Predefined CIFAR 10 classes


class Network(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3,6,5)
		self.pool = nn.MaxPool2d(2,2)
		self.conv2 = nn.Conv2d(6,16,5)
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120,84)
		self.fc3 = nn.Linear(84,10)

	def forward(self, tensor):
		tensor = self.pool(F.relu(self.conv1(tensor)))
		tensor = self.pool(F.relu(self.conv2(tensor)))
		tensor = tensor.view(-1, 16*5*5)
		tensor = F.relu(self.fc1(tensor))
		tensor = F.relu(self.fc2(tensor))
		tensor = self.fc3(tensor)
		return tensor

network = Network()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr = 0.001, momentum = 0.90, nesterov = True) 

for epoch in range(5):
	running_loss = 0.0
	for index, data in enumerate(trainloader, 0):
		inputs, labels = data
		optimizer.zero_grad() # Zero parametric gradient buffers
		outputs = network(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		running_loss+=loss.item()
		if index % 2000 == 1999:
			 print('[%d, %5d] loss: %.3f' %(epoch + 1, index + 1, running_loss / 2000)) # Log successive running loss to stdout at 2,000 minibatch intervals
			 running_loss = 0.0

print("Finished training")

dataiter = iter(testloader)
images, labels = dataiter.next()
outputs = network(images)
_, predicted = torch.max(outputs, 1) # Retrieve the latter value, namely, the column indices of the supremum along each row
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))







