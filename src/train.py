import numpy as np
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def rard_solver(x0, t_final, initial_dt=0.01, momentum=0.9):
    """
    RARD-inspired solver: a momentum-augmented solver using PyTorch.
    
    Args:
        x0: Initial state
        t_final: Final time
        initial_dt: Initial step size
        momentum: Momentum parameter
        
    Returns:
        np.array: Trajectory of (time, state) pairs
    """
    t = 0.0
    x = torch.tensor([x0], dtype=torch.float32)
    v = torch.tensor([0.0], dtype=torch.float32)  # momentum state
    traj = [(t, x.item())]
    dt = initial_dt
    min_dt, max_dt = 1e-4, 0.1
    target_error = 1e-4  # threshold for adjusting dt
    while t < t_final:
        f_val = -x  # derivative, using torch math
        mid_x = x + 0.5 * dt * (v + f_val)
        v_new = momentum * v - dt * (-mid_x)
        x_new = x + dt * v_new
        t += dt
        exact = x0 * np.exp(-t)
        est_error = abs(x_new.item() - exact)
        if est_error > target_error:
            dt = max(dt / 1.5, min_dt)
        else:
            dt = min(dt * 1.2, max_dt)
        x = x_new
        v = v_new
        traj.append((t, x.item()))
    return np.array(traj)


def dynamic_regime_solver(x0, t_final, initial_dt=0.01, momentum=0.9):
    """
    Solver that adapts to different regimes during diffusion.
    
    Args:
        x0: Initial state
        t_final: Final time
        initial_dt: Initial step size
        momentum: Momentum parameter
        
    Returns:
        np.array: Trajectory of (time, state) pairs
        list: Regime scores
    """
    t = 0.0
    x = torch.tensor([x0], dtype=torch.float32)
    v = torch.tensor([0.0], dtype=torch.float32)
    traj = [(t, x.item())]
    regime_scores = []  # record synthetic RP scores
    dt = initial_dt
    min_dt, max_dt = 1e-4, 0.1
    target_error = 1e-4
    while t < t_final:
        f_val = -x
        regime = 'memorization' if x.item() > 0.5 else 'generalization'
        exact = x0 * np.exp(-t)
        current_error = abs(x.item() - exact)
        rp_score = current_error  # simple RP score for demonstration
        regime_scores.append(rp_score)
        mid_x = x + 0.5 * dt * (v + f_val)
        if regime == 'memorization':
            dt = max(dt / 1.5, min_dt)
            correction = 0.5 * (mid_x - x)  # a placeholder correction term
        else:
            dt = min(dt * 1.2, max_dt)
            correction = 0.0
        v_new = momentum * v - dt * (-mid_x + correction)
        x_new = x + dt * v_new
        t += dt
        x = x_new
        v = v_new
        traj.append((t, x.item()))
    return np.array(traj), regime_scores


def fixed_threshold_solver(x0, t_final, dt=0.01, momentum=0.9):
    """
    Solver that uses fixed thresholds (baseline for comparison).
    
    Args:
        x0: Initial state
        t_final: Final time
        dt: Fixed step size
        momentum: Momentum parameter
        
    Returns:
        np.array: Trajectory of (time, state) pairs
        list: RP scores
    """
    t = 0.0
    x = torch.tensor([x0], dtype=torch.float32)
    v = torch.tensor([0.0], dtype=torch.float32)
    traj = [(t, x.item())]
    rp_scores = []
    while t < t_final:
        f_val = -x
        mid_x = x + 0.5 * dt * (v + f_val)
        v_new = momentum * v - dt * (-mid_x)  # no regime-adaptive correction
        x_new = x + dt * v_new
        t += dt
        x = x_new
        v = v_new
        exact = x0 * np.exp(-t)
        rp_score = abs(x.item() - exact)
        rp_scores.append(rp_score)
        traj.append((t, x.item()))
    return np.array(traj), rp_scores


def diffusion_sampling(samples, config, n_steps=100):
    """
    Diffusion sampling procedure with options for ablating components.
    
    Args:
        samples: Initial samples
        config: Configuration dictionary with flags for different components
        n_steps: Number of steps to run
        
    Returns:
        torch.Tensor: Final samples
        list: Trajectory of samples at each step
        list: RP scores at each step
    """
    dt = 0.05
    momentum_val = 0.9
    velocity = torch.zeros_like(samples)
    trajectory = []  # list of samples at each step
    rp_scores = []   # simulated RP score = mean distance to target (0,0)
    for i in range(n_steps):
        grad = -samples  
        if config['regime_adaptive']:
            correction = -0.5 * grad  
        else:
            correction = 0.0
        if config['dynamic_tuning']:
            mae = samples.abs().mean().item()
            effective_dt = dt * 0.7 if mae > 0.8 else dt * 1.1
        else:
            effective_dt = dt

        if config['momentum']:
            velocity = momentum_val * velocity + effective_dt * (grad + correction)
            update = velocity
        else:
            update = effective_dt * (grad + correction)
        samples = samples + update
        trajectory.append(samples.clone())
        rp = samples.norm(dim=1).mean().item()
        rp_scores.append(rp)
    return samples, trajectory, rp_scores
