import torch
a = torch.randn(2, 3)

a = a.to("cuda")
print(a)