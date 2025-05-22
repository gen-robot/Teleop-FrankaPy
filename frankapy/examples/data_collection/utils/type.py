import numpy as np
import torch 

def get_numpy(data, device="cpu"):
    if isinstance(data, torch.Tensor):
        if device == "cpu":
            return data.numpy()
        else:
            return data.cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError("parameter passed is not torch.tensor")

def to_numpy(data, device="cpu",):
    if isinstance(data, torch.Tensor):
        return get_numpy(data, device)
    elif isinstance(data, dict):
        return {key: to_numpy(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        try:
            return np.array([to_numpy(item, device) for item in data])
        except ValueError:
            return [to_numpy(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(to_numpy(item, device) for item in data)
    elif isinstance(data, bytes):
        return np.frombuffer(data, dtype=np.uint8)
    else:
        return data
