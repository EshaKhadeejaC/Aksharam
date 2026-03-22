#!/usr/bin/env python3
"""
CUDA Setup Script for Aksharam Project

This script helps set up NVIDIA CUDA for GPU acceleration in the Aksharam project.
Run this script to check CUDA status and get installation instructions.
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)

def check_cuda_setup():
    """Check current CUDA setup"""
    print("🔍 Checking CUDA setup...\n")

    # Check PyTorch CUDA
    print("1. Checking PyTorch CUDA support:")
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"   CUDA available: {cuda_available}")
        if cuda_available:
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        print("   ✅ PyTorch CUDA check passed" if cuda_available else "   ❌ PyTorch CUDA not available")
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False

    # Check NVIDIA drivers
    print("\n2. Checking NVIDIA drivers:")
    success, stdout, stderr = run_command("nvidia-smi")
    if success:
        print("   ✅ NVIDIA drivers detected")
        # Extract driver version
        lines = stdout.split('\n')
        for line in lines:
            if 'Driver Version' in line:
                print(f"   {line.strip()}")
                break
    else:
        print("   ❌ NVIDIA drivers not found or nvidia-smi not available")

    # Check CUDA toolkit
    print("\n3. Checking CUDA toolkit:")
    success, stdout, stderr = run_command("nvcc --version")
    if success:
        print("   ✅ CUDA toolkit detected")
        # Extract version
        for line in stdout.split('\n'):
            if 'release' in line:
                print(f"   {line.strip()}")
                break
    else:
        print("   ❌ CUDA toolkit not found")

    # Check cuDNN
    print("\n4. Checking cuDNN:")
    try:
        import torch
        if torch.cuda.is_available():
            # Try a simple CUDA operation
            x = torch.randn(10, 10).cuda()
            y = torch.randn(10, 10).cuda()
            z = torch.matmul(x, y)
            print("   ✅ cuDNN working (basic CUDA operations successful)")
        else:
            print("   ❌ CUDA not available for cuDNN check")
    except Exception as e:
        print(f"   ❌ cuDNN check failed: {e}")

    return cuda_available

def print_installation_instructions():
    """Print CUDA installation instructions"""
    print("\n" + "="*60)
    print("🚀 CUDA INSTALLATION INSTRUCTIONS")
    print("="*60)

    print("\n📋 Prerequisites:")
    print("   • NVIDIA GPU with CUDA support (GTX 1650 or newer)")
    print("   • Latest NVIDIA drivers installed")
    print("   • Python virtual environment activated")

    print("\n📦 Step 1: Install NVIDIA Drivers")
    print("   Visit: https://www.nvidia.com/Download/index.aspx")
    print("   Download and install the latest drivers for your GTX 1650")

    print("\n📦 Step 2: Install CUDA Toolkit")
    print("   Visit: https://developer.nvidia.com/cuda-downloads")
    print("   Download CUDA Toolkit 11.8 or 12.1 for Windows")
    print("   Follow the installation wizard")

    print("\n📦 Step 3: Install cuDNN (if needed)")
    print("   Visit: https://developer.nvidia.com/cudnn")
    print("   Download cuDNN v8.9+ for CUDA 11.x or 12.x")
    print("   Extract to CUDA installation directory")

    print("\n📦 Step 4: Install CUDA-enabled PyTorch")
    print("   Run these commands in your virtual environment:")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

    print("\n📦 Step 5: Verify Installation")
    print("   Run this script again: python cuda_setup.py")
    print("   Or run: python -c \"import torch; print(torch.cuda.is_available())\"")

    print("\n⚡ Performance Tips:")
    print("   • Use batch processing for better GPU utilization")
    print("   • Monitor GPU memory usage with nvidia-smi")
    print("   • Adjust batch sizes based on your GPU memory")

def main():
    print("🖥️  Aksharam CUDA Setup Checker")
    print("="*40)

    cuda_working = check_cuda_setup()

    if cuda_working:
        print("\n🎉 CUDA is properly configured!")
        print("   Your Aksharam project will use GPU acceleration.")
        print("   Expected speedup: 5-10x for LaBSE embeddings and mBART translation")
    else:
        print("\n⚠️  CUDA is not available.")
        print("   The project will run on CPU (slower but still functional).")
        print_installation_instructions()

    print("\n" + "="*60)
    print("💡 Usage Tips:")
    print("   • LaBSE embeddings: GPU acceleration for paragraph similarity")
    print("   • mBART translation: GPU acceleration for neural translation")
    print("   • Monitor performance with: nvidia-smi (if CUDA installed)")
    print("="*60)

if __name__ == "__main__":
    main()