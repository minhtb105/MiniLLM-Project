import logging
import numpy as np


class Function:
    @classmethod
    def apply(cls, *inputs):
        import sys
        
        Tensor = None
        if "src.core.tensor" in sys.modules:
            Tensor = sys.modules["src.core.tensor"].Tensor
        elif "__main__" in sys.modules:
            Tensor = sys.modules["__main__"].Tensor

        ctx = {}
        raw_inputs = [i.data if hasattr(i, "data") else i for i in inputs]
        out_data = cls.forward(ctx, *raw_inputs)
        out = Tensor(out_data)
        out.requires_grad = any(getattr(i, "requires_grad", False) for i in inputs)
        fn = cls()
        fn.ctx = ctx
        fn.parents = [i for i in inputs if isinstance(i, Tensor)]
        out.grad_fn = fn
        
        return out
        
class Add(Function):
    @staticmethod 
    def forward(ctx: dict, a, b):
        ctx["a_shape"] = a.shape
        ctx["b_shape"] = b.shape
        
        return a + b
    
    @staticmethod 
    def backward(ctx: dict, grad_output):
        return grad_output, grad_output
    
class Sub(Function):
    @staticmethod
    def forward(ctx: dict, a, b):
        ctx["a"], ctx["b"] = a, b
        return a - b
    
    @staticmethod
    def backward(ctx: dict, grad_output):
        return grad_output, -grad_output  

class Neg(Function):
    @staticmethod
    def forward(ctx: dict, a):
        return -a
    
    @staticmethod
    def backward(ctx: dict, grad_output):
        # d(-a)/da = -1
        return -grad_output
    
class Mul(Function):
    @staticmethod
    def forward(ctx: dict, a, b):
        ctx["a"], ctx["b"] = a, b
        return a * b
    
    @staticmethod
    def backward(ctx: dict, grad_output):
        return grad_output * ctx["b"], grad_output * ctx["a"]  
    
class Pow(Function):
    @staticmethod
    def forward(ctx: dict, a, power):
        ctx["a"], ctx["p"] = a, power
        return a ** power
    
    @staticmethod
    def backward(ctx: dict, grad_output):
        a, p = ctx['a'], ctx['p']
        grad_a = grad_output * p * (a ** (p - 1))
        # attempt grad wrt p if p is tensor-like; if p scalar, the caller likely won't require grad
        try:
            grad_p = grad_output * (a ** p) * np.log(a + 1e-9)
            return grad_a, grad_p
        except Exception:
            return grad_a, None
    
class Div(Function):
    @staticmethod
    def forward(ctx: dict, a, b):
        ctx["a"], ctx["b"] = a, b
        return a / b
    
    @staticmethod
    def backward(ctx: dict, grad_output):
        a, b = ctx["a"], ctx["b"]
        return grad_output * (1 / b), grad_output * (-a / (b ** 2))
    
class Tanh(Function):
    @staticmethod
    def forward(ctx: dict, a):
        out = np.tanh(a)
        ctx["out"] = out
        return out
    
    @staticmethod
    def backward(ctx: dict, grad_output):
        # d(tanh(x))/dx = 1 - tanh^2(x)
        return grad_output * (1 - ctx["out"] ** 2)
    
class GELU(Function):
    @staticmethod
    def forward(ctx: dict, a):
        # Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        c = np.sqrt(2 / np.pi)
        ctx["a"] = a
        ctx["c"] = c
        ctx["inner"] = c * (a + 0.044715 * (a ** 3))
        ctx["tanh_inner"] = np.tanh(ctx["inner"])
        out = 0.5 * a * (1 + ctx["tanh_inner"])
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx["a"]
        c = ctx["c"]
        tanh_inner = ctx["tanh_inner"]
        inner = ctx["inner"]
        # Derivative of GELU approximation
        left = 0.5 * (1 + tanh_inner)
        right = 0.5 * a * (1 - tanh_inner ** 2) * c * (1 + 3 * 0.044715 * a ** 2)
        grad = grad_output * (left + right)
        return grad
    
class Log(Function):
    @staticmethod
    def forward(ctx, a):
        ctx["a"] = a
        return np.log(a + 1e-9)  # avoid log(0)
    
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx["a"]
        return grad_output * (1 / (a + 1e-9))
    
class Sqrt(Function):
    @staticmethod
    def forward(ctx, a):
        out = np.sqrt(a)
        ctx["out"] = out
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * (0.5 / (ctx["out"] + 1e-9))

class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        out = np.exp(a)
        ctx["out"] = out
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx["out"]

class Softmax(Function):
    @staticmethod
    def forward(ctx, x):
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        out = exp_x / (exp_x.sum(axis=-1, keepdims=True) + 1e-9)
        ctx["out"] = out
        return out

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx["out"]
        grad = y * (grad_output - (grad_output * y).sum(axis=-1, keepdims=True))
        
        return grad
    
class MatMul(Function):
    @staticmethod
    def forward(ctx: dict, a, b):
        ctx["a"], ctx["b"] = a, b
        return a @ b
    
    @staticmethod
    def backward(ctx: dict, grad_output):
        a, b = ctx["a"], ctx["b"]
        grad_a = grad_output @ b.T
        grad_b = a.T @ grad_output
        
        return grad_a, grad_b
    
class ReLU(Function):
    @staticmethod
    def forward(ctx: dict, a):
        ctx["mask"] = (a > 0).astype(np.float32)
        return a * ctx["mask"]
    
    @staticmethod
    def backward(ctx: dict, grad_output):
        return grad_output * ctx["mask"]
    
class Sum(Function):
    @staticmethod
    def forward(ctx: dict, a, axis=None, keepdims=False):
        ctx["a_shape"] = a.shape
        ctx["axis"] = axis
        ctx["keepdims"] = keepdims
        
        return np.sum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, grad_output):
        # grad sum = 1
        a_shape = ctx["a_shape"]
        axis = ctx["axis"]
        keepdims = ctx["keepdims"]
        
        if not keepdims and axis is not None:
            grad_output = np.expand_dims(grad_output, axis)
        grad = np.ones(a_shape, dtype=np.float32) * grad_output
        
        return grad

class Mean(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        ctx["a_shape"] = a.shape
        ctx["axis"] = axis
        ctx["keepdims"] = keepdims

        return np.mean(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, grad_output):
        a_shape = ctx["a_shape"]
        axis = ctx["axis"]
        keepdims = ctx["keepdims"]

        if not keepdims and axis is not None:
            grad_output = np.expand_dims(grad_output, axis)

        # Gradient of mean = 1/N for each element
        scale = np.prod(a_shape) / np.prod(grad_output.shape)
        grad = np.ones(a_shape, dtype=np.float32) * (grad_output / scale)
        
        return grad
    