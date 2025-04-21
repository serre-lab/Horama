"""
Module for the Feature Accentuation optimization process.
See Hamblin et al. (2024) for more details.
"""

import torch
from tqdm import tqdm
from einops import rearrange

from .common import recorrelate_colors, get_color_correlation_svd_sqrt
from .fourier_fv import get_fft_scale, optimization_step


class SpectrumBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, spectrum_scaler):
        ctx.save_for_backward(spectrum_scaler)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (spectrum_scaler,) = ctx.saved_tensors
        return grad_output * spectrum_scaler[None, :, :], None


class SpectrumBackwardScaler(torch.nn.Module):
    """
    Acts as identity during forward pass, but scales the gradient
    by the spectrum scaler during backward pass.

    Parameters
    ----------
    spectrum_scaler : torch.Tensor
        The spectrum scaler to use during the backward pass.
    """

    def __init__(self, spectrum_scaler):
        super().__init__()
        self.register_buffer("spectrum_scaler", spectrum_scaler)

    def forward(self, x):
        return SpectrumBackwardFunction.apply(x, self.spectrum_scaler)


def fa_preconditionner(spectrum, spectrum_backward_scaler, values_range, device):
    # feature accentuation preconditioner used to convert the fourier spectrum
    # to spatial domain
    assert spectrum.shape[0] == 3

    spectrum = spectrum_backward_scaler(spectrum)

    spatial_image = torch.fft.irfft2(spectrum)
    color_recorrelated_image = recorrelate_colors(spatial_image, device)

    image = torch.sigmoid(color_recorrelated_image) * (values_range[1] - values_range[0]) + values_range[0]

    return image


def inverse_image_spectrum(image, device=None):
    # inverse of fa_preconditionner, convert image to fourier spectrum
    # by inverting our preconditionning process
    if device is None:
        device = image.device

    img_max = torch.amax(image)
    img_min = torch.amin(image)

    image = (image - img_min) / (img_max - img_min)

    eps = 1e-3
    inv_sigmoid = torch.log((image+eps) / (1.0 - image + eps))

    inv_corr_matrix = torch.linalg.pinv(get_color_correlation_svd_sqrt(device))
    inv_sigmoid = rearrange(inv_sigmoid, 'c h w -> h w c')

    inv_recorrelate = torch.matmul(inv_sigmoid.contiguous().view(-1, 3),
                                   inv_corr_matrix)
    inv_recorrelate = rearrange(
        inv_recorrelate, '(h w) c -> c h w', c=inv_sigmoid.shape[-1],
        h=inv_sigmoid.shape[0],
        w=inv_sigmoid.shape[1])

    spectrum = torch.fft.rfft2(inv_recorrelate)

    return spectrum


def accentuation(objective_function, image_seed, decay_power=1.5, total_steps=1000,
                 learning_rate=1.0, image_size=1280, model_input_size=224, noise=0.05,
                 values_range=(-2.5, 2.5), crops_per_iteration=6, box_size=(0.20, 0.25),
                 penalty=1.0, device='cuda'):
    # perform the Feature Accentuation (Hamblin & al) optimization process
    assert values_range[1] >= values_range[0]
    assert box_size[1] >= box_size[0]

    if image_seed.shape[1] != image_size:
        image_seed = torch.nn.functional.interpolate(
            image_seed.unsqueeze(0),
            size=(image_size, image_size),
            mode='bilinear', antialias=True)[0]

    spectrum = inverse_image_spectrum(image_seed, device)
    scaler = get_fft_scale(image_size, image_size, decay_power)
    spectrum_scaler = SpectrumBackwardScaler(scaler)

    spectrum = spectrum.to(device)
    spectrum.requires_grad = True
    spectrum_scaler = spectrum_scaler.to(device)

    optimizer = torch.optim.NAdam([spectrum], lr=learning_rate)
    transparency_accumulator = torch.zeros((3, image_size, image_size)).to(device)

    # tfel: recompute image_seed to avoid impossible l2 regul. (in theory we don't need it but...)
    image_seed = fa_preconditionner(spectrum, spectrum_scaler, values_range, device).detach()

    for _ in tqdm(range(total_steps)):
        optimizer.zero_grad()

        image = fa_preconditionner(spectrum, spectrum_scaler, values_range, device)
        loss, img = optimization_step(objective_function, image, box_size,
                                      noise, crops_per_iteration, model_input_size)
        loss += (img - image_seed).square().mean() * penalty
        loss.backward()
        transparency_accumulator += torch.abs(img.grad)

        optimizer.step()

    final_image = fa_preconditionner(spectrum, spectrum_scaler, values_range, device)

    return final_image, transparency_accumulator
