"""Main experiment script."""
import os
import torch
from datetime import datetime

from config.experiment_config import config
from src.preprocess import prepare_data
from src.train import SimpleNN, train_model
from src.evaluate import evaluate_model

def main():
    """Run the complete experiment pipeline."""
    print("=== Experiment Configuration ===")
    for key, value in config.__dict__.items():
        print(f"{key}: {value}")
    print("\n")
    
    print("=== Preparing Data ===")
    train_loader, test_loader = prepare_data(config)
    print(f"Training batches: {len(train_loader)}")
    print(f"Testing batches: {len(test_loader)}")
    print("\n")
    
    print("=== Initializing Model ===")
    input_size = next(iter(train_loader))[0].shape[1]
    model = SimpleNN(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers
    )
    print(model)
    print("\n")
    
    print("=== Training Model ===")
    trained_model = train_model(model, train_loader, config)
    
    # Save the trained model
    os.makedirs(config.model_dir, exist_ok=True)
    model_path = os.path.join(
        config.model_dir,
        f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    )
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    print("\n")
    
    print("=== Evaluation Results ===")
    evaluation_report = evaluate_model(trained_model, test_loader)
    print(evaluation_report)

if __name__ == "__main__":
    main()
