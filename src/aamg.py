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
                state['sum_grad_squared'] = torch.zeros_like(p.data)
                state['step'] = 0

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
                state['step'] += 1
                
                # Update sum of squared gradients (MADGRAD's adaptive scaling)
                state['sum_grad_squared'].addcmul_(grad, grad, value=1)
                
                # Compute adaptive learning rate
                adaptive_lr = group['lr'] / (state['sum_grad_squared'].sqrt() + group['eps'])
                
                # Update velocities with AggMo's multiple momentum factors
                for i, beta in enumerate(self.beta_factors):
                    if i == 0:
                        # First momentum incorporates adaptive scaling
                        state['velocities'][i].mul_(beta).addcmul_(grad, adaptive_lr, value=1 - beta)
                    else:
                        # Additional momentums for stability
                        state['velocities'][i].mul_(beta).add_(grad, alpha=1 - beta)
                
                # Compute weighted average of momentums
                avg_momentum = sum(v * (1.0 / (1.0 - beta)) 
                                 for v, beta in zip(state['velocities'], self.beta_factors))
                avg_momentum.div_(sum(1.0 / (1.0 - beta) for beta in self.beta_factors))
                
                # Update parameters
                p.data.add_(avg_momentum, alpha=-1.0)

        return loss
