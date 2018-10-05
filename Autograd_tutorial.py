import torch

x = torch.ones(2,2,requires_grad = True)

y = x+2

z = y*y*3 

out = z.mean()

print(z, out)

a = torch.randn(2,2) # Initialize a random 2x2 tensor

a = ((a*3)/(a-1))

a.requires_grad_(True) # Track computation

b = (a*a).sum()

out.backward()

print(x.grad) # Should emit a 2x2 tensor populated with 4.5

x = torch.randn(3, requires_grad = True)

y = x*2

while y.data.norm() < 1000:
	y = y*2

gradients = torch.tensor([0.1,1.0,0.8], dtype = torch.float)

y.backward(gradients)

print(x.grad)


