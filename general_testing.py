import torch
"""
[]: Size([0])
[1]: Size([1])
[1,2]: Size([2])
[[1,2],[2,3]]: Size([2,2])
[[1,2],[2,3],[4,5]]: Size([3,2])
"""
x = torch.tensor([[1], [2], [3]])
x.size()
print(x)
x = x.expand(3, 2)
print(x)
x = x.expand(3, 2)   # -1 means not changing the size of that dimension
print(x)
