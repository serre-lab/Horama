import torch
import pytest
from horama import maco, fourier, plot_maco
import matplotlib.pyplot as plt

from .common import SimpleDummyModel


@pytest.fixture
def cleanup_plot():
    yield
    plt.close('all')
    plt.ion()


def test_plot_maco(cleanup_plot):
    def objective(images): return torch.mean(model(images))
    model = SimpleDummyModel()

    img_size = 200
    image, alpha = maco(objective, total_steps=10, image_size=img_size,
                        model_input_size=100, device='cpu')

    plot_maco(image, alpha)

    fig = plt.gcf()
    assert fig is not None, "Plotting failed: no figure created"


def test_plot_fourier(cleanup_plot):
    def objective(images): return torch.mean(model(images))
    model = SimpleDummyModel()

    img_size = 200
    image, alpha = fourier(objective, total_steps=10, image_size=img_size,
                           model_input_size=100, device='cpu')

    plot_maco(image, alpha)

    fig = plt.gcf()
    assert fig is not None, "Plotting failed: no figure created"
