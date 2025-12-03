"""
check_gpu.py - Diagnostic script to check GPU/CUDA availability

Run this to diagnose why your GPU isn't being detected.
"""

import sys

print("="*70)
print("GPU/CUDA DIAGNOSTIC CHECK")
print("="*70)

# Check 1: PyTorch installation
print("\n[1/5] Checking PyTorch installation...")
try:
    import torch
    print(f"  ✓ PyTorch installed: {torch.__version__}")
except ImportError:
    print("  ✗ PyTorch not installed!")
    print("  Install with: pip install torch")
    sys.exit(1)

# Check 2: CUDA availability
print("\n[2/5] Checking CUDA availability...")
cuda_available = torch.cuda.is_available()
if cuda_available:
    print(f"  ✓ CUDA is available!")
else:
    print(f"  ✗ CUDA is NOT available")
    print(f"  PyTorch was built with CUDA: {torch.version.cuda}")

# Check 3: CUDA version
print("\n[3/5] Checking CUDA version...")
if cuda_available:
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  cuDNN version: {torch.backends.cudnn.version()}")
else:
    print(f"  PyTorch built for CUDA: {torch.version.cuda}")
    print(f"  (But CUDA runtime not detected)")

# Check 4: GPU devices
print("\n[4/5] Checking GPU devices...")
if cuda_available:
    num_gpus = torch.cuda.device_count()
    print(f"  Number of GPUs: {num_gpus}")

    for i in range(num_gpus):
        print(f"\n  GPU {i}:")
        print(f"    Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")

        # Check memory usage
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"    Memory Allocated: {allocated:.2f} GB")
        print(f"    Memory Reserved: {reserved:.2f} GB")
else:
    print("  No GPU devices found")

# Check 5: Test tensor creation
print("\n[5/5] Testing GPU tensor creation...")
if cuda_available:
    try:
        test_tensor = torch.randn(100, 100).cuda()
        print(f"  ✓ Successfully created tensor on GPU")
        print(f"  Device: {test_tensor.device}")
        del test_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ✗ Failed to create tensor on GPU: {e}")
else:
    print("  Skipped (no CUDA available)")

# Summary and recommendations
print("\n" + "="*70)
print("SUMMARY & RECOMMENDATIONS")
print("="*70)

if cuda_available:
    print("\n✓ GPU is working correctly!")
    print("\nYour transformer model should use GPU automatically.")
    print("If it's still showing 'cpu', check your transformer code.")
else:
    print("\n✗ GPU is NOT available. Possible reasons:\n")

    # Check PyTorch build
    if torch.version.cuda is None:
        print("1. PyTorch CPU-only version installed")
        print("   Solution: Install PyTorch with CUDA support")
        print("   ")
        print("   For CUDA 11.8:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("   ")
        print("   For CUDA 12.1:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    else:
        print("1. NVIDIA drivers not installed or not working")
        print("   Check with: nvidia-smi")
        print("   ")
        print("2. CUDA toolkit not installed")
        print("   Install CUDA toolkit matching your driver")
        print("   ")
        print("3. PyTorch CUDA version mismatch with driver")
        print(f"   PyTorch built for CUDA: {torch.version.cuda}")
        print("   Check your NVIDIA driver version with: nvidia-smi")

print("\n" + "="*70)

# Additional system checks
print("\nADDITIONAL SYSTEM INFO")
print("="*70)
print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")

# Try nvidia-smi
print("\nTrying nvidia-smi command...")
import subprocess
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✓ nvidia-smi output:")
        print(result.stdout[:500])  # First 500 chars
    else:
        print("✗ nvidia-smi failed")
except FileNotFoundError:
    print("✗ nvidia-smi not found (NVIDIA drivers not installed)")
except Exception as e:
    print(f"✗ Error running nvidia-smi: {e}")

print("\n" + "="*70)
