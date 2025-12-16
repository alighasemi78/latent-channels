import torch


def softmax(x, dim=-1):
    return torch.softmax(x, dim=dim)


def cosine_sim(a, b, eps=1e-8):
    # a: [d], b: [n,d] -> [n]
    a = a / (a.norm() + eps)
    b = b / (b.norm(dim=1, keepdim=True) + eps)
    return b @ a


def make_block_projectors(d: int, C: int):
    assert d % C == 0
    dc = d // C
    Ps = []
    for c in range(C):
        P = torch.zeros(d, d)
        P[c * dc : (c + 1) * dc, c * dc : (c + 1) * dc] = torch.eye(dc)
        Ps.append(P)
    return Ps


def attention(q, K, V):
    # q: [d], K: [N,d], V: [N,d]
    scores = K @ q  # [N]
    a = torch.softmax(scores, dim=0)
    o = (a[:, None] * V).sum(dim=0)  # [d]
    return o, a
