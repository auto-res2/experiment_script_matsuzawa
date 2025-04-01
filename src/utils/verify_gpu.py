import torch
import os
import sys

def verify_gpu_compatibility():
    """
    Verify that the code can run on NVIDIA Tesla T4 with 16GB VRAM.
    If running on CPU, provide information about CPU compatibility.
    """
    print("=" * 50)
    print("Hardware Compatibility Check")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        print("Note: This code is designed to run on NVIDIA Tesla T4 with 16GB VRAM.")
        print("When deployed on a system with a Tesla T4 GPU, the code will utilize GPU acceleration.")
        print("CPU execution will be significantly slower but is supported for development and testing.")
        
        print("\nTesting basic tensor operations on CPU...")
        try:
            test_tensor = torch.randn(1000, 1000)
            result = test_tensor @ test_tensor.T
            print(f"Test tensor shape: {result.shape}")
            print("CPU tensor operations test passed.")
        except Exception as e:
            print(f"CPU tensor operations test failed: {e}")
        
        print("\nCPU compatibility check completed.")
        return True  # Return True to allow execution on CPU for testing
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
    
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_memory:.2f} GB")
    
    is_t4 = "T4" in gpu_name
    has_sufficient_memory = gpu_memory >= 14.0  # Allow some margin for system usage
    
    if is_t4:
        print("GPU is Tesla T4 as required.")
    else:
        print("WARNING: GPU is not Tesla T4. The code is designed to run on Tesla T4.")
        print("The code may still run, but performance characteristics may differ.")
    
    if has_sufficient_memory:
        print("GPU has sufficient memory for the experiment.")
    else:
        print(f"WARNING: GPU memory ({gpu_memory:.2f} GB) may not be sufficient for the experiment.")
        print("The code is designed to run on a GPU with 16GB VRAM.")
        print("Reducing batch sizes or model sizes may be necessary.")
    
    try:
        print("\nTesting GPU functionality with a small tensor...")
        test_tensor = torch.zeros(1000, 1000, device='cuda')
        result = test_tensor @ test_tensor.T
        print(f"Test tensor shape: {result.shape}")
        
        del test_tensor
        del result
        torch.cuda.empty_cache()
        
        print("GPU functionality test passed.")
        
    except RuntimeError as e:
        print(f"GPU functionality test failed: {e}")
        print("The experiment may need to be adjusted to use less memory.")
        return False
    
    print("\nGPU compatibility check completed.")
    return True

if __name__ == "__main__":
    verify_gpu_compatibility()
