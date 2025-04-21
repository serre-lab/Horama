import torch
import timm
import pytest
from horama import maco, fourier, accentuation, plot_maco


@pytest.fixture
def setup_model():
    model = timm.create_model('mobilenetv3_small_050.lamb_in1k', pretrained=False).cpu().eval()
    def objective(images): return torch.mean(model(images)[:, 1])
    return model, objective


def test_fourier_timm(setup_model):
    model, objective = setup_model

    img_size = 128
    model_size = 128

    image1, alpha1 = maco(objective, total_steps=10, image_size=img_size,
                          model_input_size=model_size, device='cpu')
    plot_maco(image1, alpha1)

    assert image1.size() == (3, img_size, img_size)
    assert alpha1.size() == (3, img_size, img_size)

    image2, alpha2 = fourier(objective, total_steps=10, image_size=img_size,
                             model_input_size=model_size, device='cpu')
    plot_maco(image2, alpha2)

    assert image2.size() == (3, img_size, img_size)
    assert alpha2.size() == (3, img_size, img_size)


def test_maco_timm(setup_model):
    model, objective = setup_model

    img_size = 128
    model_size = 128

    image2, alpha2 = fourier(objective, total_steps=10, image_size=img_size,
                             model_input_size=model_size, device='cpu')
    plot_maco(image2, alpha2)

    assert image2.size() == (3, img_size, img_size)
    assert alpha2.size() == (3, img_size, img_size)


def test_fa_timm(setup_model):
    model, objective = setup_model

    img_size = 128
    model_size = 128

    image_seed = torch.rand((3, 128, 128)) * 0.01

    image3, alpha3 = accentuation(objective, image_seed, total_steps=10, image_size=img_size,
                                  model_input_size=model_size, device='cpu')
    plot_maco(image3, alpha3)

    assert image3.size() == (3, img_size, img_size)
    assert alpha3.size() == (3, img_size, img_size)


def test_fa_timm(setup_model):
    model, objective = setup_model

    img_size = 128
    model_size = 128

    # Use a deterministic seed for reproducibility
    torch.manual_seed(0)
    image_seed = torch.rand((3, img_size, img_size)) * 0.01

    # Run accentuation
    image_out, alpha = accentuation(
        objective,
        image_seed,
        total_steps=10,  # keep small for speed
        image_size=img_size,
        model_input_size=model_size,
        device='cpu'
    )

    plot_maco(image_out, alpha)

    # Check shapes
    assert image_out.size() == (3, img_size, img_size), "Output image has incorrect shape"
    assert alpha.size() == (3, img_size, img_size), "Alpha map has incorrect shape"

    # Check value range
    assert image_out.min() >= -2.5 and image_out.max() <= 2.5, "Output image values out of expected range"

    # Check that alpha is non-zero (optimization occurred)
    assert torch.sum(alpha).item() > 0, "Alpha map is zero—gradient accumulation failed"

    # Sanity check: output image isn't identical to seed
    assert not torch.allclose(image_out, image_seed, atol=1e-2), "Output too close to seed—optimization may have failed"
