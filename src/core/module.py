import logging
import numpy as np
from tensor import Tensor


class Module:
    """
    A base class similar to torch.nn.Module
    Supports:
        - Registering parameters
        - Saving and loading weights
        - Recursive access to submodles
    """
    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def add_module(self, name, module):
        self._modules[name] = module

    def add_parameter(self, name, tensor):
        assert isinstance(tensor, Tensor)
        self._parameters[name] = tensor
        
    def parameters(self):
        """
        Iterate over all parameters (including those of submodules).
        Returns a list of Tensors with requires_grad=True.
        """
        params = []
        for name, param in self._parameters.items():
            if param is not None:
                params.append((name, param))
                
        for name, module in self._modules.items():
            for sub_name, sub_param in module.parameters():
                params.append((f"{name}.{sub_name}", sub_param))
                
        return params
    
    def named_parameters(self, prefix=""):
        """
        Iterate over all parameters with their names (for saving state_dict).
        """
        for name, p in self._parameters.items():
            yield f"{prefix}{name}", p
            
        for mname, m in self._modules.items():
            for sub_name, p in m.named_parameters(f"{prefix}{mname}."):
                yield sub_name, p

    def state_dict(self):
        """
        Return a dict mapping parameter names to NumPy arrays (tensor data).
        """
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state_dict):
        """
        Load weights from dict (name -> ndarray)
        """
        for name, p in self.named_parameters():
            if name in state_dict:
                p.data = np.array(state_dict[name], dtype=np.float32)
            else:
                print(f"[Warning] Missing key in state_dict: {name}")

    def save_state_dict(self, path):
        """
        Save weights to file .npz
        """
        np.savez(path, **self.state_dict())
        logging(f"Saved model weights to {path}")

    def from_pretrained(self, path):
        """
        Load weights file .npz
        """
        state = np.load(path)
        self.load_state_dict(state)
        logging.info(f"Loaded model weights from {path}")
        
    def __setattr__(self, name, value):
        # Override __setattr__ to automatically register submodules
        if isinstance(value, Module):
            self.add_module(name, value)
        else:
            object.__setattr__(self, name, value)
            