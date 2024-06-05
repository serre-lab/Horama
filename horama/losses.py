import torch


def cosine_similarity(tensor_a, tensor_b):
    norm_dims = list(range(1, len(tensor_a.shape)))
    tensor_a = torch.nn.functional.normalize(tensor_a.float(), dim=norm_dims)
    tensor_b = torch.nn.functional.normalize(tensor_b.float(), dim=norm_dims)
    return torch.sum(tensor_a * tensor_b, dim=norm_dims)


def dot_cossim(tensor_a, tensor_b, cossim_pow=2.0):
    # see https://github.com/tensorflow/lucid/issues/116
    cosim = torch.clamp(cosine_similarity(tensor_a, tensor_b), min=1e-1) ** cossim_pow
    dot = torch.sum(tensor_a * tensor_b)
    return dot * cosim
