import torch
from tqdm import tqdm
from pixartdiffusion.parameters import *
from pixartdiffusion.noise import getβ, noise, αt
from pixartdiffusion.util import sharp_scale, draw_list
import random

# Downscales a 2D tensor exactly, using averaging
def downscale_avg(im, factor):
    X,Y = im.shape
    assert X%factor==0
    assert Y%factor==0
    X2,Y2 = X//factor, Y//factor
    im2 = torch.zeros((X2, Y2), dtype=im.dtype, device=im.device)
    for i in range(factor):
        for j in range(factor):
            im2 += im[i::factor, j::factor]
    im2 /= factor * factor
    return im2

# Takes a classifier model (and label vector) and returns a guidance function
# Classifier should take an image and a time, and make a prediction which can be compared to 'label' via MSE
def classifier_to_grad_func(classifier, label):
    def f(ims, ts):
        with torch.enable_grad():
            im_v = torch.autograd.Variable(ims, requires_grad=True)
            pred = classifier(im_v, ts)
            loss = torch.mean((label - pred)**2)
            grad = torch.autograd.grad(loss, im_v)[0]

            return grad
    return f

# Takes a CLIP model and some text and returns a guidance function
# The CLIP model's input size must be a multiple of ART_SIZE
# The input image will be shifted num_shifts times and input to CLIP each time, the gradients averaged
#  shift_range gives the max amount that it will be shifted by in pixels, AFTER it is scaled up
def clip_grad_func(clip_model, tokenized_text, num_shifts = 16, shift_range = 14):
    clip_res_mul = clip_model.visual.input_resolution // ART_SIZE

    tokenized_text = tokenized_text.to(device)
    text_encoding = clip_model.encode_text(tokenized_text)

    # (note: ts is ignored, the CLIP model is not noised)
    def f(ims, ts):
        B,_,_,_ = ims.shape
        ims = sharp_scale(ims, clip_res_mul, dims=(2,3))
        grad = torch.zeros_like(ims)
        
        possible_shifts = [(i%(2*shift_range+1) - shift_range, i//(2*shift_range+1) - shift_range) for i in range(1, (2*shift_range+1)**2)]
        random.shuffle(possible_shifts)
        possible_shifts = [(0,0)] + possible_shifts # Make sure shift of (0,0) always included

        with torch.enable_grad():
            im_v = torch.autograd.Variable(ims, requires_grad=True)

            clip_input = torch.zeros((num_shifts * B, NUM_CHANNELS, ART_SIZE*clip_res_mul, ART_SIZE*clip_res_mul), device=device, dtype=torch.float)

            # Create shifts
            for i, (shift_x, shift_y) in enumerate(possible_shifts[:num_shifts]):
                clip_input[i*B:(i+1)*B] = torch.roll(im_v, (shift_x, shift_y), (2, 3))

            # Encode each shift and get gradients
            im_encoding = clip_model.encode_image(clip_input)
            closeness = torch.sum(im_encoding * text_encoding)
            grad = torch.autograd.grad(closeness, im_v)[0] / num_shifts
                
        # Downscale grad to grad2 (e.g. 224^2 -> 32^2)
        grad2 = torch.zeros((1, 3, ART_SIZE, ART_SIZE), device=device)
        for i in range(3):
            grad2[0, i] = downscale_avg(grad[0,i], clip_res_mul)

        return grad2
    
    return f


# Given a tensor of images and a list of captions, evaluates each caption on each image
def evaluate_clip(clip_model, ims, tokenized_text):
    clip_res_mul = clip_model.visual.input_resolution // ART_SIZE
    text = tokenized_text.to(device)
    with torch.no_grad():
        ims = sharp_scale(ims, clip_res_mul, dims=(2,3))
        logits_per_im, logits_per_text = clip_model(ims, text)
        
##        if len(text_list) > 1:
##            return logits_per_im.softmax(dim=-1)
        return logits_per_im

# Get the standard deviation of sampling at time t
def getσ(t):
    # L = torch.tensor(0.0002, dtype=torch.float)
    # H = torch.tensor(0.02, dtype=torch.float)
    # return ((H - L) * t/steps + L).float()
    return getβ(t)


# Goes from x(t) -> x(t-1)
#  noise_mul: Multiplier on top of getσ(t). Higher values lead to more chaotic images
#  classifier_func: Output of clip_grad_func or classifier_to_grad_func
#  classifier_mul: Classifier multiplier
def sample_step(model, im, t, noise_mul = 8, classifier_func = None, classifier_mul = 1):
    with torch.no_grad():
        N,_,_,_ = im.shape

        z = torch.normal(torch.zeros_like(im))
        if t == 1:
            z *= 0

        ts = torch.Tensor([t]*N).to(device)

        noise = model(im, ts)

        α = 1 - getβ(t)
        new_mean = α**-0.5 * (im - (1-α)/(1-αt[t])**0.5 * noise)

        # if t==1 don't attempt to use classifier guidance
        if classifier_func != None and t != 1:
            grad = classifier_func(im, ts)
            new_mean += grad * classifier_mul * getσ(t)

        add_noise = getσ(t) * z * noise_mul
        im = new_mean + add_noise      # add random noise on

        return im

# Repeats sample_step to go from x(t) to x(0)
# im : N x C x W x H
def sample_from(model, im, t, noise_mul = 6, classifier_func = None, classifier_mul = 1):
    with torch.no_grad():
        for i in range(t, 0, -1):
            im = sample_step(model, im, i, noise_mul=noise_mul, classifier_func=classifier_func, classifier_mul=classifier_mul)
    return (im+1)/2

# 'Redraws' an image by adding some amount of noise then applying the model to remove it
# im : N x C x W x H
# jump 1: totally redraw          jump 0: no change
def redraw_im(model, im, jump=0.5, noise_mul=6, classifier_func=None, classifier_mul=1):
    im = 2 * im - 1

    N,_,_,_ = im.shape

    time = int(ACTUAL_STEPS * jump)
    t = torch.Tensor([time]*N).long().to(device)
    _, im2 = noise(im, t)
    return sample_from(model, im2, time, noise_mul=noise_mul, classifier_func=classifier_func, classifier_mul=classifier_mul)

# Sample N images from the model.
#  display_count: number of times to display intermediate result
def sample(model, N, display_count = 4, noise_mul = 6, classifier_func = None, classifier_mul = 1):
    with torch.no_grad():
        # Initial samples
        size = (N, NUM_CHANNELS, ART_SIZE, ART_SIZE)
        h = torch.normal(torch.zeros(size), 1).float().to(device)

        s = ACTUAL_STEPS // display_count if display_count != 0 else ACTUAL_STEPS*5

        for t in tqdm(range(ACTUAL_STEPS, 0, -1)):
            h = sample_step(model, h, t, noise_mul, classifier_func=classifier_func, classifier_mul=classifier_mul)

            if t % s == (s//2):
                print("ITERATION",t)
                draw_list((h+1)/2)
        
        # -1..1 -> 0..1
        return (h+1)/2

# Sorts an input list of images using CLIP
# Images must be Tensors
def CLIP_rerank(clip_model, images, tokenized_text):
    ans = evaluate_clip(clip_model, images, tokenized_text).detach().cpu()
    y = [(v,i) for i,v in enumerate(ans)]
    y = sorted(y,key=lambda a:-a[0])
    y = [i for v,i in y]
    return images[y]

if __name__=="__main__":
    # Handle command line arguments
    import argparse
    import util
    import model
    import numpy as np
    from matplotlib import image

    parser = argparse.ArgumentParser("sample.py")

    parser.add_argument("model_path", help="Path to the model.")
    parser.add_argument("num_samples", help="Number of samples.", type=int)
    parser.add_argument("-o", help="Path to save output to. The generated image will be saved as a single spritesheet .png here. If empty, does not save.", default="", nargs='?')
    parser.add_argument("-noise_mul", help="Standard deviation during sampling. Larger values lead to more chaotic samples. Default: 8.0", default=8, nargs='?', type=float)

    args = parser.parse_args()

    model = model.UNet().to(device)
    epoch = util.load_model(model, args.model_path)
    print("Loaded model from epoch", epoch)

    xs = sample(model, args.num_samples, display_count=0, noise_mul=args.noise_mul)
    sheet = util.to_drawable(xs)
    util.draw_im(sheet)

    if args.o != "":
        image.imsave(args.o, sheet)

    input("Press enter...")
    

    
