#!/usr/bin/env python3
"""
Training utilities for Iso-LWGAN experiments.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision

class SimpleEncoder(nn.Module):
    def __init__(self, z_dim):
        super(SimpleEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, z_dim)
        )
    def forward(self, x):
        return self.net(x)

class SimpleGenerator(nn.Module):
    def __init__(self, z_dim):
        super(SimpleGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, z):
        return self.net(z)

class StochasticGenerator(nn.Module):
    def __init__(self, z_dim, noise_dim=2):
        super(StochasticGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.fc1 = nn.Linear(z_dim + noise_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()

    def forward(self, z, sigma_noise=0.1):
        noise = torch.randn(z.size(0), self.noise_dim, device=z.device) * sigma_noise
        z_noise = torch.cat([z, noise], dim=1)
        out = self.relu(self.fc1(z_noise))
        out = self.relu(self.fc2(out))
        return self.fc3(out)

class MNISTEncoder(nn.Module):
    def __init__(self, z_dim):
        super(MNISTEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256), nn.ReLU(),
            nn.Linear(256, z_dim)
        )
    def forward(self, x):
        return self.net(x)

class MNISTGenerator(nn.Module):
    def __init__(self, z_dim, noise_dim=10, partial_noise=False):
        super(MNISTGenerator, self).__init__()
        self.partial_noise = partial_noise
        input_dim = z_dim + (noise_dim if partial_noise else 0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 28*28), nn.Tanh()
        )
    def forward(self, z, sigma_noise=0.1):
        if self.partial_noise:
            if z.dim() > 2:
                batch_size = z.size(0)
                z = z.view(batch_size, -1)
            noise = torch.randn(z.size(0), 10, device=z.device) * sigma_noise
            z = torch.cat([z, noise], dim=1)
        out = self.net(z)
        return out.view(-1, 1, 28, 28)

def isometric_loss(encoder, generator, x, lambda_iso):
    """
    Compute the isometric loss which penalizes the absolute difference between 
    pairwise distances in the latent space and in the generated space.
    """
    z = encoder(x)           # (B, z_dim)
    x_generated = generator(z)  # (B, 2)
    latent_dist = torch.cdist(z, z, p=2)
    gen_dist = torch.cdist(x_generated, x_generated, p=2)
    loss = torch.mean(torch.abs(latent_dist - gen_dist))
    return lambda_iso * loss

def train_iso_lwgan(lambda_iso=1.0, num_epochs=20, batch_size=128, z_dim=3, data_loader=None, 
                    n_samples=2000, device="cuda", log_dir=None, save_dir="./"):
    """
    Train the generator and encoder using an isometric loss added to a standard L2
    reconstruction loss. Saves an interpolation plot (PDF) of generated data.
    """
    print(f"\n[Experiment 1] Training Iso-LWGAN with lambda_iso = {lambda_iso}")
    
    if data_loader is None:
        from preprocess import generate_synthetic_data
        data = generate_synthetic_data(n_samples=n_samples)
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(data)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    generator = SimpleGenerator(z_dim).to(device)
    encoder = SimpleEncoder(z_dim).to(device)

    optimizer = optim.Adam(list(generator.parameters()) + list(encoder.parameters()), lr=1e-3)
    
    writer = None
    if log_dir:
        writer = SummaryWriter(log_dir=log_dir)
    
    loss_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in data_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            
            z = encoder(x)
            x_gen = generator(z)
            recon_loss = torch.mean(torch.norm(x - x_gen, dim=1))
            
            iso_loss = isometric_loss(encoder, generator, x, lambda_iso)
            
            total_loss = recon_loss + iso_loss
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
        avg_epoch_loss = epoch_loss / len(data_loader)
        loss_history.append(avg_epoch_loss)
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_epoch_loss:.4f}")
            
            if writer:
                writer.add_scalar("Loss/Total", avg_epoch_loss, epoch)
                
    with torch.no_grad():
        x_val = next(iter(data_loader))[0].to(device)
        z_val = encoder(x_val)
        x_gen_val = generator(z_val)
        latent_dist = torch.cdist(z_val, z_val, p=2)
        gen_dist = torch.cdist(x_gen_val, x_gen_val, p=2)
        avg_dist_diff = torch.mean(torch.abs(latent_dist - gen_dist)).item()
        print("Average pairwise distance difference:", avg_dist_diff)

    with torch.no_grad():
        z0 = encoder(x_val[0:1])
        z1 = encoder(x_val[1:2])
        n_interp = 10
        interp_codes = torch.stack([z0 * (1 - float(t)/(n_interp-1)) + z1 * (float(t)/(n_interp-1))
                                    for t in range(n_interp)]).to(device)
        x_interp = generator(interp_codes)
        interp_np = x_interp.cpu().numpy()
        
        plt.figure(figsize=(6, 4))
        if interp_np.shape[1] >= 2:
            plt.scatter(interp_np[:, 0], interp_np[:, 1],
                        c=np.linspace(0, 1, n_interp), cmap='viridis', s=50)
            plt.title("Latent Space Interpolation")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
        else:
            plt.plot(np.linspace(0, 1, n_interp), interp_np[:, 0], 'o-')
            plt.title("Latent Space Interpolation (1D)")
            plt.xlabel("Interpolation Step")
            plt.ylabel("Value")
        plt.tight_layout()
        pdf_filename = f"{save_dir}/latent_interpolation_lambda{lambda_iso}.pdf"
        plt.savefig(pdf_filename, bbox_inches="tight", format="pdf")
        plt.close()
        print(f"Saved latent space interpolation plot as {pdf_filename}")

    if writer:
        writer.close()
        
    return loss_history, encoder, generator

def train_stochastic_generator(sigma_noise=0.1, num_epochs=20, batch_size=128, z_dim=3, 
                               data_loader=None, n_samples=1500, device="cuda", 
                               log_dir=None, save_dir="./"):
    """
    Train the encoder and a stochastic generator that injects additive noise.
    Saves a PDF plot showing the variability induced by noise injection.
    """
    print(f"\n[Experiment 2] Training Stochastic Generator with sigma_noise = {sigma_noise}")
    
    if data_loader is None:
        from preprocess import generate_multimodal_data
        data = generate_multimodal_data(n_samples=n_samples)
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(data)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    encoder = SimpleEncoder(z_dim).to(device)
    generator = StochasticGenerator(z_dim, noise_dim=2).to(device)
    
    optimizer = optim.Adam(list(encoder.parameters()) + list(generator.parameters()), lr=1e-3)
    
    writer = None
    if log_dir:
        writer = SummaryWriter(log_dir=log_dir)
    
    loss_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in data_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            
            z = encoder(x)
            x_gen = generator(z, sigma_noise)
            recon_loss = torch.mean(torch.norm(x - x_gen, dim=1))
            recon_loss.backward()
            optimizer.step()
            
            epoch_loss += recon_loss.item()
            
        avg_epoch_loss = epoch_loss / len(data_loader)
        loss_history.append(avg_epoch_loss)
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_epoch_loss:.4f}")
            
            if writer:
                writer.add_scalar("Loss/Reconstruction", avg_epoch_loss, epoch)

    with torch.no_grad():
        sample = next(iter(data_loader))[0][0:1].to(device)
        z_fixed = encoder(sample)
        samples_list = []
        n_samples = 10
        for i in range(n_samples):
            generated = generator(z_fixed, sigma_noise).cpu().numpy()[0]
            samples_list.append(generated)
        samples_np = np.array(samples_list)
        
        plt.figure(figsize=(6,4))
        plt.scatter(samples_np[:, 0], samples_np[:, 1], c='blue', s=50)
        plt.title(f"Variation due to Noise Injection (sigma_noise={sigma_noise})")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.tight_layout()
        pdf_filename = f"{save_dir}/noise_injection_sigma{sigma_noise}.pdf"
        plt.savefig(pdf_filename, bbox_inches="tight", format="pdf")
        plt.close()
        print(f"Saved noise injection variation plot as {pdf_filename}")
    
    if writer:
        writer.close()
        
    return loss_history, encoder, generator

def train_comparison_model(model_type="Base", num_epochs=5, z_dim=20, lambda_iso=1.0, 
                         sigma_noise=0.1, data_loader=None, device="cuda", 
                         log_dir=None, save_dir="./"):
    """
    Train a model on MNIST. For model_type "Base", no isometric loss or noise is used.
    For model_type "Iso", we use both the isometric regularizer and partial noise injection.
    Loss curves are logged using TensorBoard and model checkpoints are saved.
    """
    print(f"\n[Experiment 3] Training {model_type} LWGAN on MNIST")
    
    if data_loader is None:
        from preprocess import load_mnist
        data_loader = load_mnist(batch_size=128)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    encoder = MNISTEncoder(z_dim).to(device)
    partial_noise_flag = True if model_type == "Iso" else False
    generator = MNISTGenerator(z_dim, noise_dim=10, partial_noise=partial_noise_flag).to(device)
    
    writer = None
    if log_dir:
        writer = SummaryWriter(log_dir=f"{log_dir}/{model_type}")
    
    optimizer = optim.Adam(list(encoder.parameters()) + list(generator.parameters()), lr=1e-3)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for i, (x, _) in enumerate(data_loader):
            x = x.to(device)
            optimizer.zero_grad()
            
            z = encoder(x)
            if partial_noise_flag:
                x_gen = generator(z, sigma_noise=sigma_noise)
            else:
                x_gen = generator(z, sigma_noise=0.0)
                
            x_flat = x.view(x.size(0), -1)
            x_gen_flat = x_gen.view(x_gen.size(0), -1)
            recon_loss = torch.mean(torch.norm(x_flat - x_gen_flat, dim=1))
            
            if model_type == "Iso":
                def gen_wrapper(z_input):
                    return generator(z_input, sigma_noise=sigma_noise).view(z_input.size(0), -1)
                
                iso_loss = isometric_loss(encoder, gen_wrapper, x_flat, lambda_iso)
                total_loss = recon_loss + iso_loss
            else:
                iso_loss = 0.0
                total_loss = recon_loss
                
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            batch_count += 1
            
            if i % 100 == 0 and writer:
                global_step = epoch * len(data_loader) + i
                writer.add_scalar("Loss/Total", total_loss.item(), global_step)
                writer.add_scalar("Loss/Reconstruction", recon_loss.item(), global_step)
                if model_type == "Iso":
                    writer.add_scalar("Loss/Isometric", iso_loss, global_step)
                    
        avg_epoch_loss = epoch_loss / batch_count
        print(f"[{model_type} Model] Epoch {epoch+1}/{num_epochs}: Loss = {avg_epoch_loss:.4f}")
    
    model_save_path = f"{save_dir}/{model_type}_model.pth"
    torch.save({
        'encoder': encoder.state_dict(),
        'generator': generator.state_dict(),
    }, model_save_path)
    print(f"Saved model checkpoint to {model_save_path}")
    
    if writer:
        writer.close()
    
    encoder.eval()
    generator.eval()
    with torch.no_grad():
        x_sample, _ = next(iter(data_loader))
        x_sample = x_sample.to(device)
        z0 = encoder(x_sample[0:1])
        z1 = encoder(x_sample[1:2])
        n_interp = 10
        interp_codes = torch.stack([z0 * (1 - float(t)/(n_interp-1)) + z1 * (float(t)/(n_interp-1))
                                    for t in range(n_interp)]).to(device)
        if partial_noise_flag:
            x_interp = generator(interp_codes, sigma_noise=sigma_noise)
        else:
            x_interp = generator(interp_codes, sigma_noise=0.0)
            
        grid = torchvision.utils.make_grid(x_interp, nrow=n_interp, normalize=True)
        np_grid = grid.cpu().numpy().transpose(1,2,0)
        
        plt.figure(figsize=(8, 2))
        plt.imshow(np_grid.squeeze(), cmap='gray')
        plt.title(f"{model_type} LWGAN Latent Interpolation")
        plt.axis("off")
        plt.tight_layout()
        pdf_filename = f"{save_dir}/latent_interpolation_{model_type}.pdf"
        plt.savefig(pdf_filename, bbox_inches="tight", format="pdf")
        plt.close()
        print(f"Saved MNIST latent interpolation plot as {pdf_filename}")
    
    return encoder, generator
