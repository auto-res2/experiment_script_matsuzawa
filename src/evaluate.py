"""Model evaluation utilities."""
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report

def evaluate_model(model, test_loader):
    """Evaluate the model and return detailed metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    # Generate detailed classification report
    report = classification_report(
        all_targets, 
        all_predictions,
        target_names=['Class 0', 'Class 1'],
        digits=4
    )
    
    return report
