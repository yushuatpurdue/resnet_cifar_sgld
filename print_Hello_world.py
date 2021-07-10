import torch
a = torch.zeros(4)
a.mul_(2).add_(3,2)
print(a)