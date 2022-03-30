# TODO: Convert this file into command-line arguments
#  This is bad practice leftover from when this was a notebook

import torch

# mode = RGB, HSV, GREY
MODE = "RGB"

if MODE == "RGB" or MODE == "HSV":
    NUM_CHANNELS = 3
if MODE == "GREY":
    NUM_CHANNELS = 1

ART_SIZE = 32

STEPS = 500
ACTUAL_STEPS = 490

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
