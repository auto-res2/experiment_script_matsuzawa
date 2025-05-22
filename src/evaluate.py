#!/usr/bin/env python3
"""
Evaluation module for the Adaptive Characteristic Simulation (ACS) experiments.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .preprocess import save_plot

def experiment1_evaluate(t_fixed, y_fixed, t_adaptive, y_adaptive, time_fixed, time_adaptive):
    """
    Evaluate the results of Experiment 1 and generate plots.
    
    Args:
        t_fixed: Time values from fixed step integration
        y_fixed: y values from fixed step integration
        t_adaptive: Time values from adaptive step integration
        y_adaptive: y values from adaptive step integration
        time_fixed: Computational time for fixed step integration
        time_adaptive: Computational time for adaptive step integration
    """
    print("Fixed-step integration: iterations =", len(t_fixed), "Time taken: {:.4f} s".format(time_fixed))
    print("Adaptive-step integration: iterations =", len(t_adaptive), "Time taken: {:.4f} s".format(time_adaptive))

    plt.figure(figsize=(10, 5))
    plt.plot(t_fixed, y_fixed, 'o-', label='Fixed step')
    plt.plot(t_adaptive, y_adaptive, 'x-', label='Adaptive step')
    plt.xlabel('Time')
    plt.ylabel('y')
    plt.title('Synthetic Diffusion Trajectories')
    plt.legend()
    save_plot(plt.gcf(), "synthetic_diffusion_trajectory")

def experiment2_evaluate(traj_base, traj_acs, steps):
    """
    Evaluate the results of Experiment 2 and generate plots.
    
    Args:
        traj_base: Trajectory from base SDE integration
        traj_acs: Trajectory from ACS SDE integration
        steps: Time steps for plotting
    """
    base_variance = torch.var(traj_base, dim=0).item()
    acs_variance = torch.var(traj_acs, dim=0).item()
    print("Base SDE variance:", base_variance)
    print("ACS SDE variance:", acs_variance)

    plt.figure(figsize=(10, 5))
    plt.plot(steps, traj_base.squeeze().cpu().numpy(), label='Base (Fixed noise)')
    plt.plot(steps, traj_acs.squeeze().cpu().numpy(), label='ACS (Adaptive noise)')
    plt.xlabel('Time')
    plt.ylabel('x')
    plt.title('SDE Integration with Controlled Random Perturbations')
    plt.legend()
    save_plot(plt.gcf(), "sde_integration_controlled")

def plot_error_trajectory(trajectory, title, filename):
    """
    Compute and plot the L2 norm of changes between successive steps.
    
    Args:
        trajectory: Array of trajectory values
        title: Plot title
        filename: Filename to save the plot
    """
    errors = []
    for i in range(1, len(trajectory)):
        err = torch.norm(trajectory[i] - trajectory[i - 1]).item()
        errors.append(err)
    plt.figure(figsize=(8, 4))
    plt.plot(errors, 'o-')
    plt.xlabel('Iteration')
    plt.ylabel('L2 Norm Change')
    plt.title(title)
    save_plot(plt.gcf(), filename)

def imshow(img, title, filename):
    """
    Display and save an image tensor.
    
    Args:
        img: Image tensor
        title: Plot title
        filename: Filename to save the plot
    """
    img_np = img.detach().squeeze().cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    img_np = np.transpose(img_np, (1, 2, 0))
    plt.figure(figsize=(3, 3))
    plt.imshow(img_np)
    plt.title(title)
    plt.axis('off')
    save_plot(plt.gcf(), filename)

def ddim_sample_base(model, x_init, num_steps=50, eta=0.0):
    """
    Base DDIM sampling with fixed step size for Experiment 3.
    """
    x = x_init.clone()
    trajectory = [x.cpu().detach()]
    dt = 1.0 / num_steps
    for i in range(num_steps):
        noise_pred = model(x, torch.tensor([i * dt]))
        x = x - dt * noise_pred
        trajectory.append(x.cpu().detach())
    return x, trajectory

def ddim_sample_acs(model, x_init, num_steps=50, eta=0.0, tol=1e-3):
    """
    ACS sampling: adaptive integration using variable step size for Experiment 3.
    """
    x = x_init.clone()
    trajectory = [x.cpu().detach()]
    t = 0.0
    step = 0
    dt = 1.0 / num_steps
    while t < 1.0:
        noise_pred = model(x, torch.tensor([t]))
        error = noise_pred.abs().mean().item()  # simplified error proxy
        if error > tol:
            dt_adaptive = dt * 0.5
        elif error < (tol / 4.0):
            dt_adaptive = dt * 1.5
        else:
            dt_adaptive = dt
        x = x - dt_adaptive * noise_pred
        t += dt_adaptive
        step += 1
        trajectory.append(x.cpu().detach())
        if step > num_steps * 3:
            break
    return x, trajectory

def experiment3_evaluate(x_base, traj_base, x_acs, traj_acs):
    """
    Evaluate the results of Experiment 3 and generate plots.
    
    Args:
        x_base: Final sample from base method
        traj_base: Trajectory from base method
        x_acs: Final sample from ACS method
        traj_acs: Trajectory from ACS method
    """
    print("Base DDIM sampling iterations:", len(traj_base))
    print("ACS sampling iterations:", len(traj_acs))

    plot_error_trajectory(traj_base, 'Base Method Error Trajectory', "error_trajectory_base")
    plot_error_trajectory(traj_acs, 'ACS Method Error Trajectory', "error_trajectory_acs")

    imshow(x_base, "Base Method Final Sample", "final_sample_base")
    imshow(x_acs, "ACS Final Sample", "final_sample_acs")
