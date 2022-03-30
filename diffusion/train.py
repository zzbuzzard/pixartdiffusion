# Handle command line arguments
import argparse

parser = argparse.ArgumentParser("train.py")

parser.add_argument("data_path", help="Path (e.g. 'blah/blah/*.png') to a folder containing the training data.")
parser.add_argument("-load_path", help="Path to load the model from. Leave empty to start from scratch.", default="", nargs='?')
parser.add_argument("-save_path", help="Path to save the model to during training. If empty, defaults to 'load_path' if this is non-empty, otherwise 'pix_model.pt' at your current location", default="", nargs='?')
parser.add_argument("-save_on", help="The model is saved every 'save_on' epochs.", default=25, nargs='?', type=int)
parser.add_argument("-print_on", help="Updates the loss graph every 'print_on' epochs. Pass a value <= 0 to never display the loss graph.", default=25, nargs='?', type=int)

args = parser.parse_args()

assert args.save_on > 0

if args.save_path == "":
    if args.load_path != "":
        args.save_path = args.load_path
    else:    
        print("Saving to pix_model.pt")
        args.save_path = "pix_model.pt"



import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

import model
import dset
import sample
import util
import noise
from parameters import ACTUAL_STEPS, device


learning_rate = 1e-4
batch_size = 128


# Load training data
print("Loading dataset...")
dataset = dset.PixDataset(args.data_path)
print("Loaded dataset of size", len(dataset))
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = model.UNet().to(device)
epoch = 1
if args.load_path != "":
    epoch = util.load_model(model, args.load_path)
    print("Loaded model from epoch", epoch)

# SIMPLE_LOSS defines which version of loss will be used. Should probably be set to true.
SIMPLE_LOSS = True

mse = nn.MSELoss()
if SIMPLE_LOSS:
    def loss_fn(ims, xs, ts):
        return mse(ims, xs)
else:
    def loss_fn(ims, xs, ts):
        loss = mse(ims, xs)
        α = 1 - getβ(ts)
        mul = getβ(ts)**2 / (2 * getσ(t)**2 * α * (1-αt[ts]))
        return torch.sum(mul * loss)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # betas=(0.5, 0.999)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)  # Very high γ as my dataset was small

epoch_vals = []
loss_vals  = []



# Performs one epoch of training
# Note: I was working with quite a small dataset, so it prints nothing during the epoch
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    prints = 3
    iters = size // batch_size

    # idk what you're doing but sure
    if iters == 0:
        iters = 1

    print_on = iters // prints

    model.train()

    tot_loss = 0

    for batch, X in enumerate(dataloader):
        B,_,_,_ = X.shape
        
        # 0..1 -> -1..1
        X = X * 2 - 1
        X = X.to(device)

        t = torch.rand((B,)).to(device)
        # t = t * t
        t = (t * ACTUAL_STEPS).long()

        err,Y = noise.noise(X,t)

        pred = model(Y, t)

        loss = loss_fn(pred, err, t)

        tot_loss += float(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return tot_loss / iters

plt.interactive(True)

avgs = [11, 41, 71]
alphas = [1/3, 2/3, 1]

# Training loop. Just trains forever, because why not.
while True:
    print(f"Epoch {epoch}")
    print("LR =", scheduler.get_last_lr())
    loss = train(train_loader, model, loss_fn, optimizer)

    print("Average loss:", loss)

    epoch_vals.append(epoch)
    loss_vals.append(loss)

    if args.print_on > 0 and epoch % args.print_on == 0:
        print("EPOCH", epoch)
        #ys = sample.sample(model, 4, display_count=0, noise_mul=8)
        #util.draw_list(ys)

        # Print (epoch, loss) graph, with 11-, 41- and 71-smoothed loss values.
        if len(epoch_vals) >= max(avgs):
            xs = epoch_vals
            loss_vals_np = np.array(loss_vals)
            for i in range(len(avgs)):
                ys = util.moving_average_pad(loss_vals_np, n=avgs[i])
                plt.plot(xs, ys, label=f"{avgs[i]}-smoothed", alpha=alphas[i])
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("Loss")

            plt.show(block=False)
            plt.pause(0.1)
            plt.cla()

    if epoch % args.save_on == 0:
        print("SAVING MODEL")
        util.save_model(model, epoch, args.save_path)
    
    if scheduler.get_last_lr()[0] > 1e-5:
        scheduler.step()
    
    epoch += 1

# Changed at 8435 (5e-5)
# Changed at 10625 (2e-5)
# Changed steps 500->450 at 11019  -> Loss went up! Huh... I guess the pure noise parts are easy..?
