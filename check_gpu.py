"""
check_gpu.py - GPU/CUDA Diagnostic Tool
Author: Shirssack

Run this to check if your GPU is detected by PyTorch
Usage: python check_gpu.py
"""

def check_gpu():
    """Check GPU/CUDA availability and provide recommendations."""

    print("=" * 70)
    print("GPU/CUDA DIAGNOSTIC CHECK")
    print("=" * 70)

    # Step 1: Check PyTorch
    print("\n[Step 1] Checking PyTorch installation...")
    try:
        import torch
        print(f"  ✓ PyTorch version: {torch.__version__}")
    except ImportError:
        print("  ✗ PyTorch NOT installed!")
        print("\nTo install PyTorch with CUDA support:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return

    # Step 2: Check CUDA
    print("\n[Step 2] Checking CUDA availability...")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA is available!")
        print(f"  ✓ CUDA version: {torch.version.cuda}")
    else:
        print(f"  ✗ CUDA NOT available")
        print(f"  PyTorch built for CUDA: {torch.version.cuda}")
        if torch.version.cuda is None:
            print("\n  Problem: You have CPU-only PyTorch installed")
            print("\n  Solution: Reinstall PyTorch with CUDA support")
            print("  Check your CUDA version with: nvidia-smi")
            print("  Then install matching PyTorch version:")
            print("\n  For CUDA 11.8:")
            print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print("\n  For CUDA 12.1:")
            print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        else:
            print("\n  Problem: CUDA drivers not working or not installed")
            print("  Check with: nvidia-smi")
        return

    # Step 3: Check GPU devices
    print("\n[Step 3] Checking GPU devices...")
    num_gpus = torch.cuda.device_count()
    print(f"  Number of GPUs: {num_gpus}")

    for i in range(num_gpus):
        print(f"\n  GPU {i}:")
        print(f"    Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024 ** 3)
        print(f"    Memory: {memory_gb:.2f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")

    # Step 4: Test GPU
    print("\n[Step 4] Testing GPU tensor creation...")
    try:
        test = torch.randn(10, 10).cuda()
        print(f"  ✓ Successfully created tensor on: {test.device}")
        del test
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return

    # Step 5: Check nvidia-smi
    print("\n[Step 5] Checking nvidia-smi...")
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("  ✓ nvidia-smi working")
            lines = result.stdout.split('\n')[:10]
            for line in lines:
                if line.strip():
                    print(f"    {line}")
        else:
            print("  ✗ nvidia-smi failed")
    except FileNotFoundError:
        print("  ✗ nvidia-smi not found")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n✓ GPU is working correctly!")
    print("✓ Your transformer model will use GPU automatically")
    print("\nTo train with GPU:")
    print("  python train_transformer.py --epochs 3 --batch_size 8")
    print("=" * 70)


if __name__ == "__main__":
    check_gpu()
