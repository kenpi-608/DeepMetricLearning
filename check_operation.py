from __future__ import print_function
import torch

if __name__ == '__main__':
    # Check the pytorch operation
    x = torch.rand(5, 3)
    print(x)
    print(torch.cuda.is_available())