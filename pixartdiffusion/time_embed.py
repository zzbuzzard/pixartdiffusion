import torch
import math

# Sinusoidal timestamp embedding

# must be even
TIME_DIM = 30
D = 10000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_timestamp_embedding(timestamps):
    pe = torch.zeros(len(timestamps), TIME_DIM).to(device)
    k2 = torch.arange(0, TIME_DIM // 2, dtype=torch.float).to(device)

    div_term = torch.exp(-k2 / TIME_DIM * math.log(D))

    ts = timestamps.unsqueeze(1)

    pe[:, 0::2] = torch.sin(ts * div_term)
    pe[:, 1::2] = torch.cos(ts * div_term)
    
    return pe
