from tensor import Tensor
import numpy as np


class Optimizer:
    def __init__(self, params, lr=1e-3):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        
    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
        
    def step(self):
        raise NotImplementedError
        
class SGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.v = [np.zeros_like(p.data) for p in self.params]
        
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            self.v[i] = self.momentum * self.v[i] - self.lr * p.grad
            p.data += self.v[i]
            
class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0
        
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad

            # Update biased first moment
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            # Update biased second moment
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)

            # Bias correction
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)

            # Update parameters
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)   
        
if __name__ == "__main__":
    x = Tensor(np.random.randn(10, 3), requires_grad=False)
    y_true = Tensor(np.random.randn(10, 1), requires_grad=False)

    W = Tensor(np.random.randn(3, 1), requires_grad=True)
    b = Tensor(np.zeros((1,)), requires_grad=True)
    
    model = lambda x : x.linear(W, b)
    mse = lambda pred, target: ((pred - target) * (pred - target)).mean()
    
    opt = Adam([W, b], lr=1e-2)

    for epoch in range(1000):
        # Forward
        y_pred = model(x)
        loss = mse(y_pred, y_true)

        # Backward
        opt.zero_grad()
        loss.backward()

        # Update params
        opt.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss = {loss.data:.6f}")
            