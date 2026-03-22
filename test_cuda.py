#!/usr/bin/env python3
"""
CUDA Test Script for Aksharam Project

Quick test to verify CUDA functionality in the project components.
"""

import torch
import sys

def test_pytorch_cuda():
    """Test basic PyTorch CUDA functionality"""
    print("🔥 Testing PyTorch CUDA...")

    if not torch.cuda.is_available():
        print("   ❌ CUDA not available")
        return False

    print(f"   ✅ CUDA available: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Test basic CUDA operations
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        print("   ✅ Basic CUDA operations working")
        return True
    except Exception as e:
        print(f"   ❌ CUDA operations failed: {e}")
        return False

def test_labse_cuda():
    """Test LaBSE model with CUDA"""
    print("\n🧠 Testing LaBSE CUDA...")

    try:
        from labse import LaBSEMatcher

        matcher = LaBSEMatcher()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Using device: {device}")

        # Quick test with small data
        test_texts = ["Hello world", "This is a test"]
        embeddings = matcher.get_embeddings(test_texts)
        print(f"   ✅ LaBSE embeddings generated: {embeddings.shape}")

        return True
    except ImportError:
        print("   ❌ LaBSE dependencies not available")
        return False
    except Exception as e:
        print(f"   ❌ LaBSE test failed: {e}")
        return False

def test_mbart_cuda():
    """Test mBART translation with CUDA"""
    print("\n🌐 Testing mBART CUDA...")

    try:
        from translate import MBARTTranslator

        translator = MBARTTranslator()
        device = "cuda" if translator.device == 0 else "cpu"
        print(f"   Using device: {device}")

        # Quick translation test
        result = translator.translate_text("Hello world", "en_XX", "ml_IN")
        print(f"   ✅ Translation working: {len(result)} chars")

        return True
    except ImportError:
        print("   ❌ mBART dependencies not available")
        return False
    except Exception as e:
        print(f"   ❌ mBART test failed: {e}")
        return False

def main():
    print("🧪 Aksharam CUDA Test Suite")
    print("="*35)

    all_passed = True

    # Test PyTorch CUDA
    if not test_pytorch_cuda():
        all_passed = False

    # Test LaBSE
    if not test_labse_cuda():
        all_passed = False

    # Test mBART
    if not test_mbart_cuda():
        all_passed = False

    print("\n" + "="*35)
    if all_passed:
        print("🎉 All CUDA tests passed!")
        print("   Your Aksharam project is ready for GPU acceleration.")
    else:
        print("⚠️  Some tests failed.")
        print("   Check the error messages above for details.")
        print("   The project will still work on CPU.")

    print("\n💡 Next steps:")
    print("   • Run full pipeline: python full_pipeline.py")
    print("   • Monitor performance: python cuda_monitor.py")
    print("   • Check setup: python cuda_setup.py")

if __name__ == "__main__":
    main()