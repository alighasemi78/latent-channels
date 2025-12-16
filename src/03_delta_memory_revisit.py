import torch
from utils import make_block_projectors, cosine_sim

torch.manual_seed(0)

d = 64
C = 4
Ps = make_block_projectors(d, C)

T = 500
k = 5

# A planted "topic" vector in channel 2
topic = torch.randn(d)
topic = Ps[2] @ topic
topic = topic / (topic.norm() + 1e-8)

# Memory per channel: list of deltas
M = [[] for _ in range(C)]

# Generate deltas; occasionally insert topic-related deltas in channel 2
for t in range(T):
    for c in range(C):
        delta = Ps[c] @ torch.randn(d) * 0.2
        if c == 2 and (t % 50 == 0):
            delta += topic * 1.0
        M[c].append(delta)

# Query that asks for topic
q = topic + 0.1 * (Ps[2] @ torch.randn(d))
q = q / (q.norm() + 1e-8)

# Retrieve top-k from channel 2
deltas = torch.stack(M[2], dim=0)  # [T, d]
sims = cosine_sim(q, deltas)  # [T]
topk = torch.topk(sims, k=k)

retrieved = deltas[topk.indices].sum(dim=0)
score = float(torch.dot(retrieved / (retrieved.norm() + 1e-8), topic))

print("top-k sims:", [float(x) for x in topk.values])
print("retrieved alignment with topic:", score)
