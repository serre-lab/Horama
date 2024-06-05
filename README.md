# Horama: A Compact Library for Feature Visualization Experiments

@todo add notebooks
@todo add illustration images + logo

Horama provides the implementation code for the research paper:

- *Unlocking Feature Visualization for Deeper Networks with MAgnitude Constrained Optimization* by Thomas Fel*, Thibaut Boissin*, Victor Boutin*, Agustin Picard*, Paul Novello*, Julien Colin, Drew Linsley, Tom Rousseau, Rémi Cadène, Laurent Gardes, Thomas Serre. [Read the paper on arXiv](https://arxiv.org/abs/2211.10154).

In addition, this repository introduces various feature visualization methods, including a reimagined approach to the [incredible work of the Clarity team](https://distill.pub/2017/feature-visualization/) and an implementation of [Feature Accentuation](https://arxiv.org/abs/2402.10039) from Hamblin & al. For an official reproduction of distill's work complete with comprehensive notebooks, we highly recommend [Lucent](https://github.com/greentfrapp/lucent). However, Horama focuses on **experimentation** within PyTorch, offering a compact and easily hackable codebase.

# 🚀 Getting Started with Horama

Horama requires Python 3.6 or newer and several dependencies, including Numpy. It supports both Tensorflow and Torch. Installation is straightforward with Pypi:

```bash
pip install horama
```

With Horama installed, you can dive into feature visualization. The API is designed to be intuitive across both Tensorflow and Pytorch frameworks, requiring only a few hyperparameters to get started.

Example usage:

```python
import torch
import timm
from horama import maco, fourier, plot_maco

%config InlineBackend.figure_format = 'retina'

model = timm.create_model('resnet18', pretrained=True).cuda().eval()

objective = lambda images: torch.mean(model(images)[:, 1])

image1, alpha1 = maco(objective)
plot_maco(image1, alpha1)
plt.show()

image2, alpha2 = fourier(objective)
plot_maco(image2, alpha2)
plt.show()
```

# Notebooks 

@todo: fourier, maco for various models on timm
@todo: cossim vs logits
@todo: speedup process, what parameters to change
@todo: feature inversion
@todo: feature accentuation

# Complete API

Complete API Guide
Horama's API includes the following primary functions:


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
        device='cuda')

```

When optimizing, it's crucial to fine-tune the hyperparameters. Parameters like the decay spectrum in the Fourier method significantly impact the visual output, controlling the energy distribution across frequencies. Additionally, adjust the values_range to match your model's preprocessing requirements, and ensure model_input_size matches the expected input size of your model. Typically, setting the noise parameter to about 10% of the input range yields satisfactory results.

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

A simpler and maintain implementation of the code for Tensorflow and the other feature visualization methods used in the paper come from the [Xplique toolbox](https://github.com/deel-ai/xplique). Additionally, we have created a website called the [LENS Project](https://github.com/serre-lab/Lens), which features the 1000 classes of ImageNet.

For a code faithful to the original work of the Clarity team, we highly recommend [Lucent](https://github.com/greentfrapp/lucent).

# Authors

- [Thomas Fel](https://thomasfel.fr) - thomas_fel@brown.edu, PhD Student DEEL (ANITI), Brown University