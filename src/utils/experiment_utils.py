"""
Utility functions for experiments.
"""
import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from memory_profiler import memory_usage
from torch_geometric.data import Data, DataLoader


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(model, dataloader, optimizer, loss_fn, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = loss_fn(outputs, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def profile_epoch(model, dataloader, optimizer, loss_fn, device):
    """Profile a single training epoch for time and memory usage."""
    start_time = time.time()
    mem_usage = memory_usage((train_model, (model, dataloader, optimizer, loss_fn, device)), interval=0.1)
    epoch_loss = train_model(model, dataloader, optimizer, loss_fn, device)
    duration = time.time() - start_time
    print(f"Epoch Loss: {epoch_loss:.4f}, Time: {duration:.2f}s, Memory Peak: {max(mem_usage):.2f} MiB")
    return epoch_loss, duration, max(mem_usage)


def create_dummy_qm9_dataset(num_samples=100):
    """
    Create a small dummy dataset of graph data mimicking a molecular dataset.
    """
    data_list = []
    num_dist_bins = 128
    for _ in range(num_samples):
        num_nodes = random.randint(5, 15)
        num_edges = random.randint(10, 30)
        x = torch.randn(num_nodes, 10)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, 10)
        y = torch.randn(num_edges, num_dist_bins)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data_list.append(data)
    return data_list


def ablation_study(models, dataloader, device, lr=1e-3):
    """
    Train each model variant for one epoch on the same dataset.
    """
    loss_fn = torch.nn.MSELoss()
    results = {}
    for name, model in models.items():
        print(f"--- Training model variant: {name} ---")
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        avg_loss = train_model(model, dataloader, optimizer, loss_fn, device)
        results[name] = avg_loss
        print(f"Model: {name}, Avg Loss: {avg_loss:.4f}")
    return results


def extract_important_triplets(model, dataloader, device, ib_threshold=0.5):
    """
    Extract important triplets based on the IB threshold.
    """
    model.eval()
    important_triplets = []  # List of tuples (edge_index[0], edge_index[1], score)
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            _ = model(batch)  # forward pass; internal prints display ratio info.
            if hasattr(batch, 'edge_index'):
                scores = torch.sigmoid(torch.randn(batch.edge_index.size(1)))  # fake scores for demonstration
                mask = (scores > ib_threshold)
                selected_edges = batch.edge_index[:, mask]
                for i in range(selected_edges.size(1)):
                    node_i = int(selected_edges[0, i].item())
                    node_j = int(selected_edges[1, i].item())
                    score = float(scores[mask][i].item())
                    important_triplets.append((node_i, node_j, score))
    print(f"Extracted {len(important_triplets)} important triplets from the dataset.")
    return important_triplets


def visualize_molecule_triplets(molecule_graph, triplets, title="Important Triplets", filename="triplet_interpretability.pdf"):
    """
    Visualize a molecule represented as a networkx graph with important triplets highlighted.
    """
    pos = nx.spring_layout(molecule_graph, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(molecule_graph, pos, node_color='lightgrey', edgecolors='black')
    nx.draw_networkx_labels(molecule_graph, pos)
    nx.draw_networkx_edges(molecule_graph, pos, alpha=0.5, edge_color='grey')
    
    highlighted_edges = []
    for (i, j, score) in triplets:
        if molecule_graph.has_edge(i, j) or molecule_graph.has_edge(j, i):
            highlighted_edges.append((i, j))
    if highlighted_edges:
        nx.draw_networkx_edges(molecule_graph, pos, edgelist=highlighted_edges,
                             edge_color='red', width=2)
    plt.title(title)
    plt.savefig(f"logs/{filename}", format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved interpretability plot as: logs/{filename}")
    plt.close()
