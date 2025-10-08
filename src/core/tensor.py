import numpy as np
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.random_utils import set_seed


def unbroadcast(grad, shape):
    """
    Reduce grad (np.array) to match `shape` by summing over broadcasted axes.
    Works when grad.shape >= shape (NumPy broadcasting rules).
    """
    grad = np.array(grad) # ensure ndarray
    # If shape is scalar
    if shape == ():
        return grad.sum()
    
    # 1) sum leading axes if grad has more dims than shape
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # 2) for dims where original shape ==1 but grad dim >1, sum along that axis
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] > 1:
            grad = grad.sum(axis=i, keepdims=True)
        
    # Finally, ensure shape equality
    if grad.shape != tuple(shape):
        grad = grad.reshape(shape)
        
    return grad

class Tensor:
    def __init__(self, data, requires_grad: bool=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None  # The function that created this tensor (used for backpropagation)
        
    def zero_grad(self, recursive=True):
        """
        Reset grad for this tensor (and optionally all tensors in the graph).
        If recursive=True, traverses the computational graph to clear all grads.
        """
        visited = set()
        self._zero_grad_recursive(visited) if recursive else setattr(self, "grad", None)
        
    def _zero_grad_recursive(self, visited: set):
        if self in visited:
            return
        
        visited.add(self)
              
        # Reset current grad
        if self.requires_grad:
            self.grad = None
            
        # Continue to parents
        if self.grad_fn:
            for p in self.grad_fn.parents:
                p._zero_grad_recursive(visited)
    
    def backward(self, grad_output=None):
        if grad_output is None:
            grad_output = np.ones_like(self.data)    
        self.grad = grad_output 
        
        visited = set()
        topo = []
        self._build_topo(topo, visited)
        
        for t in reversed(topo):
            if t.grad_fn is None:
                continue
            
            grad_out = t.grad
            if grad_out is None:
                grad_out = np.ones_like(t.data)
                
            grads = t.grad_fn.backward(t.grad_fn.ctx, grad_out)
            if not isinstance(grads, (tuple, list)):
                grads = (grads,) 
                
            parents = t.grad_fn.parents
            # If lengths differ, try to align
            for parent, g in zip(t.grad_fn.parents, grads):
                if parent.requires_grad:
                    # reduce gradient to parent's original shape
                    g_reduced = unbroadcast(g, parent.data.shape)
                    parent.grad = (parent.grad + g_reduced) if parent.grad is not None else g_reduced

    def _build_topo(self, topo, visited):
        if self not in visited:
            visited.add(self)
            if self.grad_fn:
                for p in self.grad_fn.parents:
                    p._build_topo(topo, visited)
                    
            topo.append(self)
        
    # convenience ops (wrap scalars to Tensor without auto requires_grad)
    def _wrap(self, other):
        return other if isinstance(other, Tensor) else Tensor(other)
        
    def __add__(self, other):
        other = self._wrap(other)
        return Add.apply(self, other)
    
    def __radd__(self, other):
        other = self._wrap(other)
        return Add.apply(other, self)
    
    def __sub__(self, other):
        other = self._wrap(other)
        return Sub.apply(self, other)
    
    def __rsub__(self, other):
        other = self._wrap(other)
        return Sub.apply(other, self)
    
    def __mul__(self, other):
        other = self._wrap(other)
        return Mul.apply(self, other)
    
    def __rmul__(self, other):
        other = self._wrap(other)
        return Mul.apply(other, self)
    
    def __pow__(self, other):
        other = self._wrap(other)
        return Pow.apply(self, other)
    
    def __rpow__(self, other):
        other = self._wrap(other)
        return Pow.apply(other, self)
    
    def __truediv__(self, other):
        other = self._wrap(other)
        return Div.apply(self, other)
    
    def __rtruediv__(self, other):
        other = self._wrap(other)
        return Div.apply(other, self)
    
    def __neg__(self):
        return Neg.apply(self)
    
    def pow(self, exponent):
        """
        Element-wise power: self ** exponent
        """
        exponent = self._wrap(exponent)
        return Pow.apply(self, exponent)
    
    def _rpow__(self, exponent):
        exponent = self._wrap(exponent)
        return Pow.apply(exponent, self)
        
    def tanh(self):
        return Tanh.apply(self)

    def log(self):
        return Log.apply(self)

    def sqrt(self):
        return Sqrt.apply(self)

    def exp(self):
        return Exp.apply(self)

    def softmax(self):
        return Softmax.apply(self)
    
    def matmul(self, other):
        other = self._wrap(other)
        return MatMul.apply(self, other)
    
    def linear(self, W, b=None):
        out = self.matmul(W)
        if b is not None:
            out = out + b
        return out
    
    def relu(self):
        return ReLU.apply(self)
    
    def relu(self):
        return GELU.apply(self)
    
    def sum(self, axis=None, keepdims=False):
        return Sum.apply(self, axis, keepdims)
            
    def mean(self, axis=None, keepdims=False):
        return Mean.apply(self, axis, keepdims)
            
    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, requires_grad={self.requires_grad})"
    
class Function:
    @classmethod
    def apply(cls, *inputs):
        ctx = {}    
        raw_inputs = [i.data if isinstance(i, Tensor) else i for i in inputs]
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

def grad_check(func, inputs, eps=1e-3, tol=1e-2):
    # Ensure no stale grads
    for x in inputs:
        x.zero_grad(recursive=False)
        
    out = func(*inputs)
    # zero grads again to be safe (if func used some intermediate tensors)
    for x in inputs:
        x.zero_grad(recursive=False)
        
    out.backward()
    analytic_grads = [x.grad.copy() for x in inputs]

    num_grads = []
    for i, x in enumerate(inputs):
        grad = np.zeros_like(x.data)
        it = np.nditer(x.data, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig = x.data[idx]
            
            x.data[idx] = orig + eps
            plus = func(*inputs).data.copy()
            
            x.data[idx] = orig - eps
            minus = func(*inputs).data.copy()
            
            x.data[idx] = orig
            
            # If plus or minus contains NaN, raise informative error
            if np.isnan(plus).any() or np.isnan(minus).any():
                raise ValueError(f"NaN encountered during numeric check for param {i} at index {idx}.\n"
                                f"plus={plus}, minus={minus}\n"
                                "Likely domain error (e.g. negative**non-integer, log(<=0), sqrt(neg)).")
            
            grad[idx] = np.sum((plus - minus) / (2 * eps))
            it.iternext()

        num_grads.append(grad)
        
    for i, (a, n) in enumerate(zip(analytic_grads, num_grads)):
        diff = np.abs(a - n).max()
        print(f"Param {i}: max|analytic - numeric| = {diff:.6e}")
        assert diff < tol, f"Gradient check failed for param {i} (diff={diff})"

    print("Gradient check passed!")

def test_all_grads():
    # --- 1. Add ---
    x = Tensor(np.random.randn(3, 3), requires_grad=True)
    y = Tensor(np.random.randn(3, 3), requires_grad=True)
    grad_check(lambda a, b: (a + b).sum(), [x, y])

    # --- 2. Sub ---
    grad_check(lambda a, b: (a - b).sum(), [x, y])

    # --- 3. Mul ---
    grad_check(lambda a, b: (a * b).sum(), [x, y])

    # --- 4. Div ---
    grad_check(lambda a, b: (a / (b + 2)).sum(), [x, y])  

    # --- 5. Pow ---
    # make base positive and bounded away from zero, exponent also shifted
    grad_check(lambda a, b: ((a * 0.5 + 2.0) ** (b * 0.5 + 2.0)).sum(), [x, y])

    # --- 6. Tanh ---
    grad_check(lambda a: a.tanh().sum(), [x])

    # --- 7. Log ---
    grad_check(lambda a: (a * 0.5 + 1.0).log().sum(), [x])  # avoid log(0)

    # --- 8. Sqrt ---
    grad_check(lambda a: ((a * 0.5 + 2).sqrt()).sum(), [x])

    # --- 9. Exp ---
    grad_check(lambda a: a.exp().sum(), [x])

    # --- 10. Softmax ---
    grad_check(lambda a: a.softmax().sum(), [x])

    # --- 11. MatMul ---
    A = Tensor(np.random.randn(2, 3), requires_grad=True)
    B = Tensor(np.random.randn(3, 4), requires_grad=True)
    grad_check(lambda a, b: a.matmul(b).sum(), [A, B])

    # --- 12. ReLU ---
    grad_check(lambda a: a.relu().sum(), [x])

    # --- 13. Sum ---
    grad_check(lambda a: a.sum(), [x])

    print("\nAll gradient checks passed successfully!")
 
def test_broadcast_grad():
    print("==== Test: Broadcasting Addition ====")
    x = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)   # (1,3)
    y = Tensor(np.array([[10.0], [20.0]]), requires_grad=True)    # (2,1)
    z = x + y  # broadcast to (2,3)
    z.backward(np.ones_like(z.data))  # all ones gradient

    # Expected results
    expected_x_grad = np.array([[2.0, 2.0, 2.0]])  # sum over broadcasted dimension
    expected_y_grad = np.array([[3.0], [3.0]])

    assert x.grad.shape == (1, 3), f"x.grad shape mismatch: {x.grad.shape}"
    assert y.grad.shape == (2, 1), f"y.grad shape mismatch: {y.grad.shape}"
    assert np.allclose(x.grad, expected_x_grad), f"x.grad mismatch:\n{x.grad}\n!=\n{expected_x_grad}"
    assert np.allclose(y.grad, expected_y_grad), f"y.grad mismatch:\n{y.grad}\n!=\n{expected_y_grad}"
    print("Broadcasting add passed!")

    # -------------------------------------------------------------------
    print("\n==== Test: Scalar Left Add ====")
    a = Tensor(2.0, requires_grad=True)
    b = 3 + a
    b.backward()
    assert np.allclose(a.grad, 1.0), f"a.grad mismatch: {a.grad} != 1"
    print("Scalar left add passed!")

    # -------------------------------------------------------------------
    print("\n==== Test: Broadcasting Multiply ====")
    p = Tensor(np.array([[2.0], [3.0]]), requires_grad=True)   # (2,1)
    q = Tensor(np.array([[4.0, 5.0, 6.0]]), requires_grad=True) # (1,3)
    r = p * q  # broadcast to (2,3)
    r.backward(np.ones_like(r.data))

    expected_p_grad = np.array([[15.0], [15.0]])  # sum over columns: 4+5+6
    expected_q_grad = np.array([[5.0, 5.0, 5.0]])  # sum over rows: 2+3

    assert p.grad.shape == (2, 1), f"p.grad shape mismatch: {p.grad.shape}"
    assert q.grad.shape == (1, 3), f"q.grad shape mismatch: {q.grad.shape}"
    assert np.allclose(p.grad, expected_p_grad), f"p.grad mismatch:\n{p.grad}\n!=\n{expected_p_grad}"
    assert np.allclose(q.grad, expected_q_grad), f"q.grad mismatch:\n{q.grad}\n!=\n{expected_q_grad}"
    print("Broadcasting multiply passed!")

    print("\nAll broadcasting gradient tests passed!\n")
    
if __name__ == "__main__":
    set_seed(42)
    
    # Run all gradient checks
    test_all_grads()
    
    test_broadcast_grad()
    
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.mean()
    y.backward()

    print("x.grad =")
    print(x.grad)
    