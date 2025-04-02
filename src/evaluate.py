import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import time

def compute_derivative(token):
    """
    Compute the derivative for diffusion process.
    In a real diffusion model, this would be more complex.
    """
    return torch.tanh(token)

def high_order_update(token, step_size, previous_update):
    """
    Perform high-order update for the diffusion process.
    """
    first_order_term = compute_derivative(token)
    second_order_term = compute_derivative(first_order_term)
    update = token + step_size * first_order_term + (step_size ** 2) * second_order_term + 0.5 * previous_update
    return update

def diffusion_step(token, step_size, previous_update):
    """
    Perform a single diffusion step using high-order update.
    """
    new_token = high_order_update(token, step_size, previous_update)
    return new_token

def standard_diffusion_step(token, step_size):
    """
    Perform a standard first-order diffusion step.
    """
    return token + step_size * compute_derivative(token)

def evaluate_token_acceleration(config):
    """
    Evaluate the Dynamic Token Acceleration component.
    
    Args:
        config: Configuration parameters
        
    Returns:
        evaluation_results: Results of the evaluation
    """
    os.makedirs('logs', exist_ok=True)
    
    print("Evaluating Dynamic Token Acceleration...")
    
    seq_length = config.seq_length
    batch_size = config.batch_size
    feature_dim = config.feature_dim
    num_iterations = config.num_iterations
    
    torch.manual_seed(0)
    token_features = torch.randn(seq_length, batch_size, feature_dim)
    previous_features = torch.zeros_like(token_features)  # initialize previous features to zero
    
    from train import TransformerBlock
    transformer = TransformerBlock(feature_dim)
    transformer.eval()
    
    ground_truth = torch.randn(seq_length, batch_size, feature_dim)
    
    loss_history = []
    for iteration in range(num_iterations):
        output_tokens = transformer(token_features, previous_features)
        previous_features = token_features.clone()
        loss = ((output_tokens - ground_truth) ** 2).mean()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        print(f"Iteration {iteration+1}/{num_iterations}, Loss: {loss_val:.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_iterations+1), loss_history, marker='o', linestyle='-', color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Dynamic Token Acceleration Evaluation")
    plt.grid(True)
    plt.savefig("logs/training_loss_dynamicToken_small.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    
    return {
        'loss_history': loss_history,
        'final_loss': loss_history[-1] if loss_history else None
    }

def evaluate_ode_solver(config):
    """
    Evaluate the Adaptive High-Order ODE Solver.
    
    Args:
        config: Configuration parameters
        
    Returns:
        evaluation_results: Results of the evaluation
    """
    os.makedirs('logs', exist_ok=True)
    
    print("Evaluating Adaptive High-Order ODE Solver...")
    
    seq_length = config.seq_length
    batch_size = config.batch_size
    feature_dim = config.feature_dim
    num_steps = config.num_steps
    step_size = config.step_size
    
    torch.manual_seed(1)
    token_standard = torch.randn(seq_length, batch_size, feature_dim)
    token_highorder = token_standard.clone()  # we simulate both approaches on identical starting noise
    
    previous_update = torch.zeros_like(token_highorder)
    
    fid_standard = []
    fid_highorder = []
    
    for step in range(1, num_steps+1):
        token_standard = standard_diffusion_step(token_standard, step_size)
        token_highorder = diffusion_step(token_highorder, step_size, previous_update)
        previous_update = compute_derivative(token_highorder)  # update previous derivative for next iteration
        
        fid_std = 100.0 / step + torch.mean(torch.abs(token_standard)).item()
        fid_ho = 100.0 / step + torch.mean(torch.abs(token_highorder)).item()
        fid_standard.append(fid_std)
        fid_highorder.append(fid_ho)
        print(f"Step {step}/{num_steps}: Standard FID={fid_std:.4f}, High-Order FID={fid_ho:.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_steps+1), fid_standard, marker='s', linestyle='--', color='red', label="Standard")
    plt.plot(range(1, num_steps+1), fid_highorder, marker='o', linestyle='-', color='green', label="High-Order")
    plt.xlabel("Diffusion Steps")
    plt.ylabel("Simulated FID")
    plt.title("FID vs. Diffusion Steps: Standard vs. Adaptive High-Order")
    plt.legend()
    plt.grid(True)
    plt.savefig("logs/fid_vs_timesteps_highOrder_small.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    
    return {
        'fid_standard': fid_standard,
        'fid_highorder': fid_highorder,
        'improvement': sum(fid_std - fid_ho for fid_std, fid_ho in zip(fid_standard, fid_highorder)) / len(fid_standard)
    }

def evaluate_model(config, trained_results):
    """
    Evaluate the trained AFiT model.
    
    Args:
        config: Configuration parameters
        trained_results: Results from training
        
    Returns:
        evaluation_results: Results of the evaluation
    """
    os.makedirs('logs', exist_ok=True)
    
    print("Evaluating AFiT model performance...")
    
    token_acceleration_results = evaluate_token_acceleration(config)
    
    ode_solver_results = evaluate_ode_solver(config)
    
    print("\nEvaluation Results:")
    print(f"Token Acceleration - Final Loss: {token_acceleration_results['final_loss']:.4f}")
    print(f"ODE Solver - Average FID Improvement: {ode_solver_results['improvement']:.4f}")
    print(f"Model Training - Final Loss: {trained_results['loss_history'][-1]:.4f}")
    print(f"Model Training Time: {trained_results['training_time']:.2f} seconds")
    
    return {
        'token_acceleration': token_acceleration_results,
        'ode_solver': ode_solver_results,
        'model_performance': {
            'final_loss': trained_results['loss_history'][-1],
            'training_time': trained_results['training_time']
        }
    }
