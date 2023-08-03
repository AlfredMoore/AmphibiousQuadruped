from torch.distributions import Normal
from torch import tensor

mu = tensor([0,0,0,0]).reshape(1,-1)
sigma = tensor([1,1,1,1]).reshape(1,-1)

print(mu)