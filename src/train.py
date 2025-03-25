#!/usr/bin/env python
"""
Training module for the HEDA experiments.
This module contains functions to run the sampling/training processes.
"""
import torch
import time
import numpy as np
from utils import compute_error, base_method_update, heda_update

def run_convergence_experiment(method_update, state_init, max_iter=100, tol=1e-3, 
                               method_name="Method", hyperparams=None):
    """
    Run a convergence experiment with a given update function.
    Logs the error at each iteration and tracks the trajectory.
    
    Args:
        method_update: Update function (base_method_update or heda_update)
        state_init: Initial state tensor
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        method_name: Name of the method for logging
        hyperparams: Dictionary of hyperparameters for the update function
    
    Returns:
        errors: List of errors at each iteration
        states: Array of states at each iteration
    """
    # Allow hyperparameters to override default update settings
    hyperparams = hyperparams if hyperparams is not None else {}
    state = state_init.clone().detach()
    errors = []
    # Save trajectory (as numpy, and later we compute averages)
    states = [state.cpu().detach().numpy()]
    prev_state = state.clone()
    
    for i in range(max_iter):
        # Update state using the provided method (pass hyperparams as needed)
        state = method_update(state, **hyperparams)
        error = compute_error(prev_state, state)
        errors.append(error)
        prev_state = state.clone()
        states.append(state.cpu().detach().numpy())
        print(f"[{method_name}] Iteration {i+1}: Error = {error:.6f}")
        if error < tol:
            print(f"{method_name} converged in {i+1} iterations (error={error:.6f}).")
            break
    else:
        print(f"{method_name} did not converge within {max_iter} iterations.")
    
    return errors, np.array(states)

def run_sampling_with_timing(method_update, state_init, max_iter=100, method_name="Method", hyperparams=None):
    """
    Run a sampling experiment for a given method.
    Measures wall-clock time per iteration along with peak GPU memory (if using cuda).
    
    Args:
        method_update: Update function (base_method_update or heda_update)
        state_init: Initial state tensor
        max_iter: Maximum number of iterations
        method_name: Name of the method for logging
        hyperparams: Dictionary of hyperparameters for the update function
    
    Returns:
        total_time: Total wall-clock time
        time_per_iter: List of times per iteration
        peak_memory: Peak GPU memory usage (in MB)
    """
    hyperparams = hyperparams if hyperparams is not None else {}
    state = state_init.clone().detach()
    time_per_iter = []
    
    # If CUDA is available, reset peak memory stats.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    start_wall = time.time()
    for i in range(max_iter):
        iter_start = time.time()
        state = method_update(state, **hyperparams)
        iter_end = time.time()
        elapsed = iter_end - iter_start
        time_per_iter.append(elapsed)
        print(f"[{method_name}] Iteration {i+1}: Time = {elapsed:.6f} sec")
    
    total_time = time.time() - start_wall
    peak_memory = torch.cuda.max_memory_allocated()/(1024**2) if torch.cuda.is_available() else None

    print(f"{method_name}: Total wall-clock time = {total_time:.2f} sec, Peak memory = "
          f"{peak_memory:.2f} MB (if using cuda)" if peak_memory is not None else "")
    
    return total_time, time_per_iter, peak_memory

def evaluate_method(method_update, hyperparams, state_init, max_iter=100, tol=1e-3):
    """
    Evaluate a given method for one set of hyperparameters.
    
    Args:
        method_update: Update function (base_method_update or heda_update)
        hyperparams: Dictionary of hyperparameters
        state_init: Initial state tensor
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
    
    Returns:
        iter_count: Number of iterations until convergence
        final_error: Final error value
    """
    # We create an inner update function that binds the hyperparameters.
    def update_with_hyper(state, **unused):
        return method_update(state, **hyperparams)
    
    errors, _ = run_convergence_experiment(update_with_hyper, state_init, max_iter=max_iter, 
                                           tol=tol, method_name="EvalTrial")
    final_error = errors[-1] if errors else None
    iter_count = len(errors)
    
    return iter_count, final_error
