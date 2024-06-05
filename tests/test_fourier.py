import torch
from horama import fourier

from .common import SimpleDummyModel


def test_fourier():
    def objective(images): return torch.mean(model(images))
    model = SimpleDummyModel()

    img_size = 200
    image, alpha = fourier(objective, total_steps=10, image_size=img_size,
                           model_input_size=100, device='cpu')

    assert image.size() == (3, img_size, img_size)
    assert alpha.size() == (3, img_size, img_size)
