<div align="center">
    <img src="assets/banner.png" width="75%" alt="Horama logo" />
</div>

<div align="center" style="margin-top: 10px;">
    <a href="#"><img src="https://img.shields.io/badge/Python-3.7–3.10-blue"></a>
    <a href="https://github.com/serre-lab/Horama/actions/workflows/lint.yml">
        <img alt="Lint Status" src="https://github.com/serre-lab/Horama/actions/workflows/lint.yml/badge.svg">
    </a>
    <a href="https://github.com/serre-lab/Horama/actions/workflows/tox.yml">
        <img alt="Tox Test Status" src="https://github.com/serre-lab/Horama/actions/workflows/tox.yml/badge.svg">
    </a>
    <a href="https://github.com/serre-lab/Horama/actions/workflows/publish.yml">
        <img alt="PyPI Publish Status" src="https://github.com/serre-lab/Horama/actions/workflows/publish.yml/badge.svg">
    </a>
    <a href="https://pepy.tech/project/horama">
        <img alt="Downloads" src="https://static.pepy.tech/badge/horama">
    </a>
    <a href="#"><img src="https://img.shields.io/badge/License-MIT-green"></a>
</div>

---

Horama is a compact library designed for Feature Visualization experiments, initially providing the implementation code for the research paper [Maco](https://arxiv.org/abs/2211.10154), developed mainly to support exploratory research.

This repository includes:

- **Maco** — implementation of the method described in [our NeurIPS 2023 paper](https://arxiv.org/abs/2211.10154).
- **Fourier-based visualizations** — inspired by [Distill’s Feature Visualization](https://distill.pub/2017/feature-visualization/).
- **Feature Accentuation** — from [Hamblin et al., 2024](https://arxiv.org/abs/2402.10039).

While Horama builds on ideas from tools like [Lucent](https://github.com/greentfrapp/lucent), it focuses on **flexibility** and **extensibility** in PyTorch. It is *not* intended as a strict reproduction of Distill’s work -- only Maco is officially reproduced here.

<div align="center">
    <strong>Quick start notebook:</strong><br>
    <a href="https://colab.research.google.com/drive/1TFYbmAgx-gbkLA4eY3lQbbZ-iPSBLiA2?usp=sharing" target="_blank">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Google Colab">
    </a>
</div>

## 🚀 Getting Started

Horama requires **Python 3.7–3.10** and works with **PyTorch** models. Installation is available via PyPI:

```bash
pip install horama
```
Once installed, you can generate feature visualizations with minimal setup. The API is consistent across all methods and designed for easy experimentation.
Example using timm:

```python
import torch
import timm
import matplotlib.pyplot as plt
from horama import maco, fourier, plot_maco

model = timm.create_model('resnet18', pretrained=True).cuda().eval()
objective = lambda images: torch.mean(model(images)[:, 1])
# run maco
image1, alpha1 = maco(objective, device='cuda')
plot_maco(image1, alpha1)
plt.show()
# run fourier
image2, alpha2 = fourier(objective, device='cuda')
plot_maco(image2, alpha2)
plt.show()
```

💡 Be sure to match device='cuda' or 'cpu' depending on where your model is loaded.

## Notebooks 📓

We provide a set of Colab notebooks to help you get started and explore different visualization techniques in Horama.

- [Starter Notebook (Colab)](https://colab.research.google.com/drive/1TFYbmAgx-gbkLA4eY3lQbbZ-iPSBLiA2?usp=sharing) – quick overview and usage examples
- Feature Inversion – *Coming soon*
- Feature Accentuation – *Coming soon*

If you're new to feature visualization or just want a fast way to test Horama, the starter notebook is a great place to begin.


## API Reference

Horama provides three main feature visualization functions. All three share a common structure and are designed for simple, composable use.

```python
maco(objective_function,
     total_steps=1000,
     learning_rate=1.0,
     image_size=1000,
     model_input_size=224,
     noise=0.1,
     values_range=(-2.5, 2.5),
     crops_per_iteration=6,
     box_size=(0.20, 0.25),
     penalty=1.0,
     device='cuda')

fourier(objective_function,
        decay_power=1.5,
        total_steps=1000,
        learning_rate=1.0,
        image_size=1000,
        model_input_size=224,
        noise=0.1,
        values_range=(-2.5, 2.5),
        crops_per_iteration=6,
        box_size=(0.20, 0.25),
        penalty=1.0,
        device='cuda')

accentuation(objective_function,
             image_seed,
             decay_power=1.5,
             total_steps=1000,
             learning_rate=1.0,
             image_size=1000,
             model_input_size=224,
             noise=0.05,
             values_range=(-2.5, 2.5),
             crops_per_iteration=6,
             box_size=(0.20, 0.25),
             penalty=1.0,
             device='cuda')
```
### Notes

- `objective_function(images) → scalar` defines what you're visualizing. It typically targets a class score, a neuron or a direction.
- `values_range` should match your model's expected input scale. For most timm models: `(-2.5, 2.5)` is usually fine.
- `box_size` controls crop scale: `(min_ratio, max_ratio)`.
- `model_input_size` must match your model's expected input size (e.g. `224`).

When optimizing, it's crucial to fine-tune the hyperparameters. Parameters like the decay spectrum in the Fourier method significantly impact the visual output, controlling the energy distribution across frequencies. Additionally, adjust the values_range to match your model's preprocessing requirements, and ensure model_input_size matches the expected input size of your model.

# Citation

```
@article{fel2023maco,
      title={Unlocking Feature Visualization for Deeper Networks with MAgnitude Constrained Optimization},
      author={Thomas, Fel and Thibaut, Boissin and Victor, Boutin and Agustin, Picard and Paul, Novello and Julien, Colin and Drew, Linsley and Tom, Rousseau and Rémi, Cadène and Laurent, Gardes and Thomas, Serre},
      journal={Advances in Neural Information Processing Systems (NeurIPS)},
      year={2023},
}
```

# Additional Resources

For a simpler and maintained implementation of the code for TensorFlow and the other feature visualization methods used in the paper, refer to the [Xplique toolbox](https://github.com/deel-ai/xplique). Additionally, we have created a website called the [LENS Project](https://github.com/serre-lab/Lens), which features the 1000 classes of ImageNet.

For code faithful to the original work of the Clarity team, we highly recommend [Lucent](https://github.com/greentfrapp/lucent).


# Authors

- [Thomas Fel](https://thomasfel.me) – tfel@g.harvard.edu
  Work done during my PhD at ANITI & Brown University, under the (great) supervision of [Thomas Serre](https://serre-lab.clps.brown.edu/).
