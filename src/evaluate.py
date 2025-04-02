import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def evaluate_experiment1(x0, t_final):
    """
    Evaluate Experiment 1: Accelerated Coarse Exploration via Momentum-Augmented ODE Solvers.
    
    Args:
        x0: Initial state
        t_final: Final time
        
    Returns:
        dict: Results including error metrics and trajectories
    """
    from train import rard_solver
    
    def f(t, x):
        return -x

    t_eval = np.linspace(0, t_final, 100)
    sol = solve_ivp(f, [0, t_final], [x0], method='RK45', t_eval=t_eval)
    t_rk = sol.t
    x_rk = sol.y[0]
    error_rk = np.abs(x_rk - x0 * np.exp(-t_rk))
    
    traj = rard_solver(x0, t_final)
    t_rard = traj[:,0]
    x_rard = traj[:,1]
    error_rard_custom = np.abs(x_rard - x0 * np.exp(-t_rard))
    
    return {
        't_rk': t_rk,
        'x_rk': x_rk,
        'error_rk': error_rk,
        't_rard': t_rard,
        'x_rard': x_rard,
        'error_rard_custom': error_rard_custom
    }


def evaluate_experiment2(x0, t_final):
    """
    Evaluate Experiment 2: Regime-Specific Correction with Dynamic Step-Size Tuning.
    
    Args:
        x0: Initial state
        t_final: Final time
        
    Returns:
        dict: Results including regime metrics and trajectories
    """
    from train import dynamic_regime_solver, fixed_threshold_solver
    
    traj_dynamic, scores_dynamic = dynamic_regime_solver(x0, t_final)
    traj_fixed, scores_fixed = fixed_threshold_solver(x0, t_final, dt=0.01)

    times_dynamic = traj_dynamic[:,0]
    xs_dynamic = traj_dynamic[:,1]
    errors_dynamic = np.abs(xs_dynamic - x0 * np.exp(-times_dynamic))

    times_fixed = traj_fixed[:,0]
    xs_fixed = traj_fixed[:,1]
    errors_fixed = np.abs(xs_fixed - x0 * np.exp(-times_fixed))
    
    return {
        'times_dynamic': times_dynamic,
        'errors_dynamic': errors_dynamic,
        'times_fixed': times_fixed,
        'errors_fixed': errors_fixed,
        'scores_dynamic': scores_dynamic,
        'scores_fixed': scores_fixed
    }


def evaluate_experiment3(n_samples=500, n_steps=100):
    """
    Evaluate Experiment 3: End-to-End Ablation Study of RARD Components.
    
    Args:
        n_samples: Number of samples
        n_steps: Number of steps
        
    Returns:
        dict: Results including ablation metrics and trajectories
    """
    from train import diffusion_sampling
    from preprocess import prepare_initial_samples
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    config_full = {'momentum': True, 'dynamic_tuning': True, 'regime_adaptive': True}
    config_no_dynamic = {'momentum': True, 'dynamic_tuning': False, 'regime_adaptive': True}
    config_no_regime  = {'momentum': True, 'dynamic_tuning': True, 'regime_adaptive': False}

    init_samples = prepare_initial_samples(n_samples, 2)

    final_full, traj_full, rp_full = diffusion_sampling(init_samples.clone(), config_full, n_steps=n_steps)
    final_no_dynamic, traj_no_dynamic, rp_no_dynamic = diffusion_sampling(init_samples.clone(), config_no_dynamic, n_steps=n_steps)
    final_no_regime, traj_no_regime, rp_no_regime = diffusion_sampling(init_samples.clone(), config_no_regime, n_steps=n_steps)
    
    final_mae_full = torch.abs(traj_full[-1]).mean().item()
    final_mae_no_dynamic = torch.abs(traj_no_dynamic[-1]).mean().item()
    final_mae_no_regime = torch.abs(traj_no_regime[-1]).mean().item()
    
    return {
        'rp_full': rp_full,
        'rp_no_dynamic': rp_no_dynamic,
        'rp_no_regime': rp_no_regime,
        'final_mae_full': final_mae_full,
        'final_mae_no_dynamic': final_mae_no_dynamic,
        'final_mae_no_regime': final_mae_no_regime
    }
