import io
import torch
import pickle

def state_dict_to_bytes(state_dict):
    """
    Serialize a PyTorch state_dict to bytes.
    """
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()

def bytes_to_state_dict(data):
    """
    Deserialize bytes to a PyTorch state_dict.
    """
    buffer = io.BytesIO(data)
    # weights_only=True is safer, available in newer torch versions
    try:
        return torch.load(buffer, weights_only=True)
    except TypeError:
         # Fallback for older torch versions
        return torch.load(buffer)

def parameters_to_bytes(parameters):
    """
    Convert a list of numpy arrays (or tensors) to bytes.
    Useful if not sending the full state_dict but just weights.
    For this project we primarily use state_dict.
    """
    buffer = io.BytesIO()
    torch.save(parameters, buffer)
    return buffer.getvalue()

def bytes_to_parameters(data):
    """
    Convert bytes back to what was saved (list of tensors/arrays).
    """
    buffer = io.BytesIO(data)
    try:
        return torch.load(buffer, weights_only=True)
    except TypeError:
        return torch.load(buffer)
