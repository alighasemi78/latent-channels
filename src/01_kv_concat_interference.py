import torch
from utils import attention

torch.manual_seed(0)

d = 64
n1, n2 = 64, 64

K1 = torch.randn(n1, d)
V1 = torch.randn(n1, d)
K2 = torch.randn(n2, d)
V2 = torch.randn(n2, d)

# Make query slightly aligned with agent 1 keys
q = K1.mean(dim=0) + 0.2 * torch.randn(d)
q = q / q.norm()

K = torch.cat([K1, K2], dim=0)
V = torch.cat([V1, V2], dim=0)

o, a = attention(q, K, V)

a1 = a[:n1]
a2 = a[n1:]

o1 = (a1[:, None] * V1).sum(dim=0)
o2 = (a2[:, None] * V2).sum(dim=0)

print("sum(attn agent1) =", float(a1.sum()))
print("sum(attn agent2) =", float(a2.sum()))
print("||o1||/||o2|| =", float(o1.norm() / (o2.norm() + 1e-8)))
print("||o|| =", float(o.norm()))
