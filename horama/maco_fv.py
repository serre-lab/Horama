import os
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.datasets.utils import download_url

from .common import optimization_step, standardize, recorrelate_colors

MACO_SPECTRUM_URL = ("https://storage.googleapis.com/serrelab/loupe/"
                     "spectrums/imagenet_decorrelated.npy")
MACO_SPECTRUM_FILENAME = 'spectrum_decorrelated.npy'


def init_maco_buffer(image_shape, std_deviation=1.0):
    # initialize the maco buffer with a random phase and a magnitude template
    spectrum_shape = (image_shape[0], image_shape[1] // 2 + 1)
    # generate random phase
    random_phase = torch.randn(
        3, *spectrum_shape, dtype=torch.float32) * std_deviation

    # download magnitude template if not exists
    if not os.path.isfile(MACO_SPECTRUM_FILENAME):
        download_url(MACO_SPECTRUM_URL, root=".",
                     filename=MACO_SPECTRUM_FILENAME)

    # load and resize magnitude template
    magnitude = torch.tensor(
        np.load(MACO_SPECTRUM_FILENAME), dtype=torch.float32)
    magnitude = F.interpolate(magnitude.unsqueeze(
        0), size=spectrum_shape, mode='bilinear', align_corners=False, antialias=True)[0]

    return magnitude, random_phase


def maco_preconditioner(magnitude_template, phase, values_range, device):
    # apply the maco preconditioner to generate spatial images from magnitude and phase
    # tfel: check why r exp^(j theta) give slighly diff results
    standardized_phase = standardize(phase)
    complex_spectrum = torch.complex(torch.cos(standardized_phase) * magnitude_template,
                                     torch.sin(standardized_phase) * magnitude_template)

    # transform to spatial domain and standardize
    spatial_image = torch.fft.irfft2(complex_spectrum)
    spatial_image = standardize(spatial_image)

    # recorrelate colors and adjust value range
    color_recorrelated_image = recorrelate_colors(spatial_image, device)
    final_image = torch.sigmoid(
        color_recorrelated_image) * (values_range[1] - values_range[0]) + values_range[0]
    return final_image


def maco(objective_function, total_steps=1000, learning_rate=1.0, image_size=1280,
         model_input_size=224, noise=0.05, values_range=(-2.5, 2.5),
         crops_per_iteration=6, box_size=(0.20, 0.25),
         device='cuda'):
    # perform the maco optimization process
    assert values_range[1] >= values_range[0]
    assert box_size[1] >= box_size[0]

    magnitude, phase = init_maco_buffer(
        (image_size, image_size), std_deviation=1.0)
    magnitude = magnitude.to(device)
    phase = phase.to(device)
    phase.requires_grad = True

    optimizer = torch.optim.NAdam([phase], lr=learning_rate)
    transparency_accumulator = torch.zeros(
        (3, image_size, image_size)).to(device)

    for _ in tqdm(range(total_steps)):
        optimizer.zero_grad()

        # preprocess and compute loss
        img = maco_preconditioner(magnitude, phase, values_range, device)
        loss, img = optimization_step(
            objective_function, img, box_size, noise, crops_per_iteration, model_input_size)

        loss.backward()
        # get dL/dx to update transparency mask
        transparency_accumulator += torch.abs(img.grad)
        optimizer.step()

    final_image = maco_preconditioner(magnitude, phase, values_range, device)

    return final_image, transparency_accumulator
