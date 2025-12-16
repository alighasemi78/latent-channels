import torch
from utils import attention, make_block_projectors

torch.manual_seed(0)

d = 64
C = 4
Ps = make_block_projectors(d, C)

# Two agents, assign them different channels
c1, c2 = 0, 1
n1, n2 = 64, 64

K1 = torch.randn(n1, d)
K2 = torch.randn(n2, d)

V1 = torch.randn(n1, d) @ Ps[c1].T  # force into channel c1
V2 = torch.randn(n2, d) @ Ps[c2].T  # force into channel c2

# Query tries to attend to both
q = K1.mean(dim=0) + K2.mean(dim=0) + 0.1 * torch.randn(d)
q = q / q.norm()

K = torch.cat([K1, K2], dim=0)
V = torch.cat([V1, V2], dim=0)

o, a = attention(q, K, V)

# Decompose output energy by channels
energies = []
leaks = []
for j in range(C):
    P = Ps[j]
    oj = P @ o
    energies.append(float(oj.norm() ** 2))
    leak = (torch.eye(d) - P) @ oj
    leaks.append(float(leak.norm() ** 2 / oj.norm() ** 2))

print("channel energies:", energies)
print("total energy:", float(o.norm() ** 2))
print("fraction in ch0:", energies[0] / (sum(energies) + 1e-8))
print("fraction in ch1:", energies[1] / (sum(energies) + 1e-8))
print("leakages:", leaks)
