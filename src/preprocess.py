import torch
import torchvision
import torchvision.transforms as transforms

def get_testloader(batch_size=32, num_workers=2, download=True):
    """
    Load CIFAR-10 test dataset and return dataloader
    
    Args:
        batch_size: Batch size for the dataloader
        num_workers: Number of workers for the dataloader
        download: Whether to download the dataset if not already downloaded
        
    Returns:
        testloader: DataLoader for the CIFAR-10 test set
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=download, 
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return testloader
