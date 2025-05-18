import numpy as np
import torch


def get_model_weights(model):
    """Get model weights as numpy arrays."""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_model_weights(model, weights):
    """Set model weights from numpy arrays."""
    state_dict = model.state_dict()
    for key, val in zip(state_dict.keys(), weights):
        state_dict[key] = torch.tensor(val.astype(np.float32))
    model.load_state_dict(state_dict)


def top_k_sparsify(tensor, k_ratio=0.01):
    """Keep only top k_ratio values (by absolute value), rest = 0."""
    if tensor.size == 0:
        return tensor

    flat = tensor.flatten()
    k = max(1, int(len(flat) * k_ratio))  # Ensure at least 1 element
    values, indices = torch.topk(torch.abs(torch.from_numpy(flat)), k, largest=True)
    sparse_flat = np.zeros_like(flat)
    sparse_flat[indices.numpy()] = flat[indices.numpy()]
    return sparse_flat.reshape(tensor.shape)


def get_topk_deltas(model, base_weights, k_ratio=0.01):
    """Calculate and sparsify weight deltas."""
    new_weights = get_model_weights(model)
    deltas = [new - base for new, base in zip(new_weights, base_weights)]
    sparse_deltas = [top_k_sparsify(delta, k_ratio) for delta in deltas]

    # Calculate actual sparsity for logging
    total_elements = sum(d.size for d in deltas)
    non_zero = sum(np.count_nonzero(d) for d in sparse_deltas)
    print(f"Sparsity: {100 * (1 - non_zero / total_elements):.2f}%")

    return sparse_deltas


def apply_deltas(base_weights, deltas):
    """Apply deltas to base weights."""
    return [base + delta for base, delta in zip(base_weights, deltas)]