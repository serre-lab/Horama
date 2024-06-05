import numpy as np
import matplotlib.pyplot as plt
import torch


def to_numpy(tensor):
    # Ensure tensor is on CPU and convert to NumPy
    return tensor.detach().cpu().numpy()


def check_format(arr):
    # ensure numpy array and move channels to the last dimension
    # if they are in the first dimension
    if isinstance(arr, torch.Tensor):
        arr = to_numpy(arr)
    if arr.shape[0] == 3:
        return np.moveaxis(arr, 0, -1)
    return arr


def normalize(image):
    # normalize image to 0-1 range
    image = np.array(image, dtype=np.float32)
    image -= image.min()
    image /= image.max()
    return image


def clip_percentile(img, percentile=0.1):
    # clip pixel values to specified percentile range
    return np.clip(img, np.percentile(img, percentile), np.percentile(img, 100-percentile))


def show(img, **kwargs):
    # display image with normalization and channels in the last dimension
    img = check_format(img)
    img = normalize(img)

    plt.imshow(img, **kwargs)
    plt.axis('off')


def plot_maco(image, alpha, percentile_image=1.0, percentile_alpha=80):
    # visualize image with alpha mask overlay after normalization and clipping
    image, alpha = check_format(image), check_format(alpha)
    image = clip_percentile(image, percentile_image)
    image = normalize(image)

    # mean of alpha across channels, clipping, and normalization
    alpha = np.mean(alpha, -1, keepdims=True)
    alpha = np.clip(alpha, None, np.percentile(alpha, percentile_alpha))
    alpha = alpha / alpha.max()

    # overlay alpha mask on the image
    plt.imshow(np.concatenate([image, alpha], -1))
    plt.axis('off')
