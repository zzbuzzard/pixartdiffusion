import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import image
from matplotlib import colors
from pixartdiffusion.parameters import *

# Loads an image into a numpy array
# Returns a float array of shape ART_SIZE x ART_SIZE x NUM_CHANNELS, normalised to 0..1
def load_im(path : str):
    im = Image.open(path).convert("RGB")
    im = np.asarray(im).astype(np.float32) / 255

    if im.shape != (ART_SIZE, ART_SIZE, NUM_CHANNELS):
        raise Exception("Invalid image size: expected " + str((ART_SIZE, ART_SIZE, NUM_CHANNELS)) + " but received " + str(im.shape))

    if MODE == "HSV":
        im = colors.rgb_to_hsv(im)

    # (H, W, C) -> (C, H, W)
    im = np.transpose(im, (2, 0, 1))
    
    if MODE == "GREY":
        im = greyscale(im)
    
    return im

# RGB -> greyscale
def greyscale(im):
    if im.shape != (3, ART_SIZE, ART_SIZE):
        raise Exception("Invalid image size: expected " + str((3, ART_SIZE, ART_SIZE)) + " but received " + str(im.shape))
    
    x = 0.2126 * im[0,:,:] + 0.7152 * im[1,:,:] + 0.0722 * im[2,:,:]
    return x.reshape((1, ART_SIZE, ART_SIZE))

# (1, ART_SIZE, ART_SIZE) -> (3, ART_SIZE, ART_SIZE), by copying
def grey_to_col(im):
    if im.shape != (1, ART_SIZE, ART_SIZE):
        raise Exception("Invalid image size: expected " + str((1, ART_SIZE, ART_SIZE)) + " but received " + str(im.shape))

    return np.repeat(im, 3, axis=0)

# Flips horizontally
def flip_hor(im):
    return np.flip(im, axis=2)

# Flips vertically
def flip_ver(im):
    return np.flip(im, axis=1)

# Adds change_amt to the hue of every pixel
def change_hue(im, change_amt):
    # (C, H, W) -> (H, W, C)
    im = np.transpose(im, (1, 2, 0))
    if MODE == "HSV":
        im[:,:,0] = (im[:,:,0] + change_amt) % 1
    else:
        hsv = colors.rgb_to_hsv(im)
        hsv[:,:,0] = (hsv[:,:,0] + change_amt) % 1
        im = colors.hsv_to_rgb(hsv)

    # (H, W, C) -> (C, H, W)
    return np.transpose(im, (2, 0, 1))

# Shifts an image by (x,y), filling the empty space with white
def shift(im, x, y):
    # (C, H, W) -> (H, W, C)
    im = np.transpose(im, (1, 2, 0))

    im = np.roll(im, x, axis=1)
    im = np.roll(im, y, axis=0)

    if MODE=="RGB":
      fill = np.array([1,1,1])
    if MODE=="HSV":
      fill = np.array([0,0,1])
    if MODE=="GREY":
      fill = np.array([1])

    # Fill the shifted edge in white
    if x>0:
        im[:,:x,:]=fill
    if x<0:
        im[:,x:,:]=fill
    if y>0:
        im[:y,:,:]=fill
    if y<0:
        im[y:,:,:]=fill

    # (H, W, C) -> (C, H, W)
    return np.transpose(im, (2, 0, 1))

# Prepares an image, as a 0..1 numpy array, for drawing
def draw_mod(im):
  im = np.transpose(im, (1, 2, 0))

  if MODE == "GREY":
      im = grey_to_col(im)
  if MODE == "HSV":
      im = colors.hsv_to_rgb(im)
  return im

# Given a tensor or numpy array with shape (B x C x W x H), draws all images using matplotlib
def draw_list(outputs):
    sheet = to_drawable(outputs)
    draw_im(sheet)

# Draws an image, given as a numpy array, as long as it is compatible with plt.imshow
def draw_im(im):
    plt.grid(False);plt.axis('off')
    plt.imshow(im)
    plt.show(block=False)
    plt.pause(0.1)

# Given a tensor or numpy array with shape (B x C x W x H), creates a spritesheet ready for drawing/downloading/etc
def to_drawable(outputs, fix_width=None, fix_height=None, scale=1):
    if type(outputs) == torch.Tensor:
        outputs = outputs.detach().cpu().numpy()
    outputs = [draw_mod(i) for i in outputs]

    sheet = make_spritesheet(outputs, fix_width=fix_width, fix_height=fix_height)
    if scale != 1:
        sheet = sharp_scale(sheet, scale)

    sheet = np.clip(sheet, 0, 1)
    return sheet
    
# Creates a spritesheet from images with size 'size'
def make_spritesheet(ims, fix_width=None, fix_height=None):
    assert fix_width is None or fix_height is None, "Canot fix both width and height!"
    
    size, size_, _ = ims[0].shape
    assert size == size_, "Spritesheet: input images must be square"

    # Calculate (width, height) of spritesheet
    if fix_width is not None:
        X = fix_width
        Y = len(ims)//X
        while X*Y < len(ims): Y+=1
    elif fix_height is not None:
        Y = fix_height
        X = len(ims)//Y
        while X*Y < len(ims): X+=1
    else:
        Y = int(len(ims) ** 0.5)
        X = len(ims) // Y
        while X*Y < len(ims): X+=1

    final = np.zeros((Y*size, X*size, 3), dtype=np.float32)

    ind = 0
    for im in ims:
        x = ind // X
        y = ind % X

        fx = size * x
        fy = size * y

        ind += 1

        final[fx:fx+size, fy:fy+size, :] = im
    
    return final

# Scales the input up by the given integer factor with no interpolation
# Note: dims must be a tuple with exactly 2 values, both in [0,len(im.shape)-1]
def sharp_scale(im, factor, dims=(0,1)):
    a,b=dims
    if type(im) == np.ndarray:
        return np.repeat(np.repeat(im, factor, axis=a), factor, axis=b)
    # else assume Tensor
    return torch.repeat_interleave(torch.repeat_interleave(im, factor, dim=a), factor, dim=b)

# Saves the model to the specified path
def save_model(model, epoch, path):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            }, path)

# Loads the model from the specified path
def load_model(model, path):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    
    model.train()

    return epoch

# Moving window average for a 1D array. n <= len(a)
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=np.float32)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Moving window average for a 1D array. n <= len(a)
# Pads the resulting array to the size of the input array (copies first and last element)
def moving_average_pad(a, n=3):
    assert n%2==1
    app = n//2
    a = moving_average(a, n)
    a0 = [a[0]]*app
    an = [a[-1]]*app
    a = np.concatenate((a0,a,an))
    return a
