import torch
from horama import accentuation
from horama.feature_accentuation import inverse_image_spectrum, fa_preconditionner, SpectrumBackwardScaler
from horama.fourier_fv import get_fft_scale

from .common import SimpleDummyModel


def test_fa():
    def objective(images): return torch.mean(model(images))
    model = SimpleDummyModel()

    image_seed = torch.rand((3, 20, 20))

    img_size = 200
    image, alpha = accentuation(objective, image_seed, total_steps=10, image_size=img_size,
                                model_input_size=100, device='cpu')

    assert image.size() == (3, img_size, img_size)
    assert alpha.size() == (3, img_size, img_size)


def test_inv_fa():
    img = torch.rand((3, 20, 20))
    img_min, img_max = torch.amin(img), torch.amax(img)

    scaler = get_fft_scale(20, 20, 1.5)
    spec_scaler = SpectrumBackwardScaler(scaler)

    img_bis = fa_preconditionner(inverse_image_spectrum(img), spec_scaler, (-img_min, img_max), 'cpu')

    assert torch.allclose(img, img_bis, atol=1e-2)
