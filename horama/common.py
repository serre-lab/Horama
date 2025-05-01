from functools import lru_cache

import torch
from torchvision.ops import roi_align
import torch.nn.functional as F


def standardize(tensor):
    # standardizes the tensor to have 0 mean and unit variance
    tensor = tensor - torch.mean(tensor)
    tensor = tensor / (torch.std(tensor) + 1e-4)
    return tensor


@lru_cache(maxsize=8)
def get_color_correlation_svd_sqrt(device):
    return torch.tensor(
        [[0.56282854, 0.58447580, 0.58447580],
         [0.19482528, 0.00000000, -0.19482528],
         [0.04329450, -0.10823626, 0.06494176]],
        dtype=torch.float32, device=device
    )


def recorrelate_colors(image, device):
    # recorrelates the colors of the images
    assert len(image.shape) == 3

    # tensor for color correlation svd square root
    color_correlation_svd_sqrt = get_color_correlation_svd_sqrt(device)

    permuted_image = image.permute(1, 2, 0).contiguous()
    flat_image = permuted_image.view(-1, 3)

    recorrelated_image = torch.matmul(flat_image, color_correlation_svd_sqrt)
    recorrelated_image = recorrelated_image.view(permuted_image.shape).permute(2, 0, 1)

    return recorrelated_image


def optimization_step(objective_function, image, box_size, noise_level,
                      number_of_crops_per_iteration, model_input_size):
    # performs an optimization step on the generated image
    # pylint: disable=C0103
    assert box_size[1] >= box_size[0]
    assert len(image.shape) == 3

    device = image.device
    image.retain_grad()

    # generate random boxes
    x0 = 0.5 + torch.randn((number_of_crops_per_iteration,), device=device) * 0.15
    y0 = 0.5 + torch.randn((number_of_crops_per_iteration,), device=device) * 0.15
    delta_x = torch.rand((number_of_crops_per_iteration,),
                         device=device) * (box_size[1] - box_size[0]) + box_size[1]
    delta_y = delta_x

    boxes = torch.stack([torch.zeros((number_of_crops_per_iteration,), device=device),
                         x0 - delta_x * 0.5,
                         y0 - delta_y * 0.5,
                         x0 + delta_x * 0.5,
                         y0 + delta_y * 0.5], dim=1) * image.shape[1]

    cropped_and_resized_images = roi_align(image.unsqueeze(
        0), boxes, output_size=(model_input_size*2, model_input_size*2)).squeeze(0)

    # add normal and uniform noise for better robustness
    cropped_and_resized_images.add_(torch.randn_like(cropped_and_resized_images) * noise_level)
    cropped_and_resized_images.add_(
        (torch.rand_like(cropped_and_resized_images) - 0.5) * noise_level)

    cropped_and_resized_images = F.interpolate(cropped_and_resized_images, size=(model_input_size, model_input_size),
                                               mode='bicubic', align_corners=True, antialias=True)

    # compute the score and loss
    score = objective_function(cropped_and_resized_images)
    loss = -score

    return loss, image
