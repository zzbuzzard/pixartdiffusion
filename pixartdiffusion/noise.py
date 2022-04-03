import torch
from pixartdiffusion.parameters import STEPS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getβ(t):
    L = torch.tensor(0.001, dtype=torch.float)
    H = torch.tensor(0.018, dtype=torch.float)

    return ((H - L) * t/STEPS + L).float()

# Returns (actual noise, noised images)
def noise(xs, ts):
    assert xs.shape[0] == ts.shape[0], f"Number of images ({xs.shape[0]}) must match number of timestamps ({ts.shape[0]})"
    assert ts.dtype == torch.long, "Times must have long datatype" # required for indexing

    βs = getβ(ts)
    ε = torch.normal(torch.zeros_like(xs), 1).to(device)
    α = αt[ts]

    noised = torch.sqrt(α)[:,None,None,None] * xs + torch.sqrt(1-α)[:,None,None,None] * ε

    return ε, noised

αt = torch.zeros(STEPS+1, dtype=torch.float).to(device)
t = 1
for i in range(STEPS+1):
    t *= 1-getβ(i)
    αt[i] = t
