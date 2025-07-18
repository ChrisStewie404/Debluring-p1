import torch.nn as nn
import torch
import numpy as np
m = nn.InstanceNorm2d(2)
def np_InstanceNorm(npt:np.array):
    desc = 1/ (np.var(npt,axis=(2,3),keepdims=True)+1e-5) ** 0.5
    exp = np.mean(npt,axis=(2,3),keepdims=True)
    return (npt - exp) *desc

input = torch.randn(1,2,2,2)
np_input = input.numpy()
np_output = np_InstanceNorm(np_input)
output = m(input)
print(output)
print(np_output)