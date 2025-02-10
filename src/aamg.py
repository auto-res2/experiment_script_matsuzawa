import torch
from torch.optim import Optimizer

class AAMG(Optimizer):
    """
    Adaptive Aggregate Momentum Gradient (AAMG) Optimizer
    Combines MADGRAD's adaptivity with AggMo's momentum aggregation.
    """

    def __init__(self, params, lr=1e-2, momentum=0.9, weight_decay=0, eps=1e-6, beta_factors=[0.9, 0.99]):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps)
        super(AAMG, self).__init__(params, defaults)

        self.beta_factors = beta_factors
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['velocities'] = [torch.zeros_like(p.data) for _ in beta_factors]

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                state = self.state[p]
                
                # Update velocities
                for i, beta in enumerate(self.beta_factors):
                    state['velocities'][i].mul_(beta).add_(grad, alpha=1 - beta)
                
                # Compute average momentum
                avg_momentum = sum(v for v in state['velocities']) / len(self.beta_factors)
                
                # Update parameters
                p.data.add_(avg_momentum, alpha=-group['lr'])

        return loss
