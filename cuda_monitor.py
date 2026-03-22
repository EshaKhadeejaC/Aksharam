#!/usr/bin/env python3
"""
CUDA Performance Monitor for Aksharam Project

Monitors GPU usage and performance metrics during ML operations.
"""

import torch
import time
import psutil
import GPUtil
from contextlib import contextmanager

class CUDAMonitor:
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.cuda_available else 0

    def get_system_info(self):
        """Get basic system and GPU information"""
        info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total / (1024**3),  # GB
            "cuda_available": self.cuda_available,
            "gpu_count": self.gpu_count,
        }

        if self.cuda_available:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB

        return info

    def get_gpu_stats(self):
        """Get current GPU statistics"""
        if not self.cuda_available:
            return None

        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    "gpu_utilization": gpu.load * 100,
                    "gpu_memory_used": gpu.memoryUsed,
                    "gpu_memory_total": gpu.memoryTotal,
                    "gpu_memory_free": gpu.memoryFree,
                    "gpu_temperature": gpu.temperature
                }
        except:
            pass

        # Fallback to PyTorch info
        return {
            "gpu_memory_allocated": torch.cuda.memory_allocated(0) / (1024**3),  # GB
            "gpu_memory_reserved": torch.cuda.memory_reserved(0) / (1024**3),   # GB
        }

    @contextmanager
    def monitor_operation(self, operation_name="Operation"):
        """Context manager to monitor an operation"""
        if not self.cuda_available:
            print(f"⚠️  {operation_name}: CUDA not available, running on CPU")
            start_time = time.time()
            yield
            end_time = time.time()
            print(".2f")
            return

        print(f"🚀 {operation_name}: Starting on GPU ({torch.cuda.get_device_name(0)})")

        # Record initial stats
        initial_stats = self.get_gpu_stats()
        start_time = time.time()

        try:
            yield
        finally:
            end_time = time.time()
            final_stats = self.get_gpu_stats()

            duration = end_time - start_time
            print(".2f")

            if initial_stats and final_stats:
                memory_used = final_stats.get("gpu_memory_used", 0) - initial_stats.get("gpu_memory_used", 0)
                print(".1f")
                if "gpu_utilization" in final_stats:
                    print(".1f")

def benchmark_embedding_generation():
    """Benchmark LaBSE embedding generation"""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        monitor = CUDAMonitor()

        # Sample paragraphs
        paragraphs = [
            "This is a sample English paragraph for testing.",
            "Another paragraph with different content.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models can be accelerated with GPUs.",
            "Natural language processing involves understanding text."
        ] * 10  # 50 paragraphs total

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer('sentence-transformers/LaBSE', device=device)

        with monitor.monitor_operation("LaBSE Embedding Generation"):
            embeddings = model.encode(paragraphs, convert_to_numpy=True, show_progress_bar=False)

        print(f"   Generated embeddings shape: {embeddings.shape}")
        return True

    except ImportError:
        print("❌ Sentence transformers not installed")
        return False
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False

def benchmark_translation():
    """Benchmark mBART translation"""
    try:
        from translate import MBARTTranslator

        monitor = CUDAMonitor()

        # Sample text
        text = "This is a sample English text for translation testing. " * 5

        translator = MBARTTranslator()

        with monitor.monitor_operation("mBART Translation"):
            result = translator.translate_text(text, src_lang="en_XX", tgt_lang="ml_IN")

        print(f"   Translated text length: {len(result)} characters")
        return True

    except ImportError:
        print("❌ Translation modules not available")
        return False
    except Exception as e:
        print(f"❌ Translation benchmark failed: {e}")
        return False

def main():
    print("🎯 Aksharam CUDA Performance Monitor")
    print("="*45)

    monitor = CUDAMonitor()
    info = monitor.get_system_info()

    print("\n📊 System Information:")
    print(f"   CPU Cores: {info['cpu_count']}")
    print(f"   Memory: {info['memory_total']:.1f} GB")
    print(f"   CUDA Available: {info['cuda_available']}")
    print(f"   GPU Count: {info['gpu_count']}")

    if monitor.cuda_available:
        print(f"   GPU Name: {info.get('gpu_name', 'Unknown')}")
        print(f"   GPU Memory: {info.get('gpu_memory_total', 0):.1f} GB")

    print("\n🔬 Running Benchmarks...")
    print("-" * 30)

    # Benchmark embedding generation
    success = benchmark_embedding_generation()
    if success:
        print("   ✅ LaBSE embedding benchmark completed")
    else:
        print("   ❌ LaBSE embedding benchmark failed")

    # Benchmark translation
    success = benchmark_translation()
    if success:
        print("   ✅ mBART translation benchmark completed")
    else:
        print("   ❌ mBART translation benchmark failed")

    print("\n💡 Tips for Optimal Performance:")
    print("   • Use batch processing for multiple texts")
    print("   • Monitor GPU memory usage during processing")
    print("   • Adjust model parameters based on GPU capabilities")
    print("   • Consider using mixed precision (FP16) for faster processing")

if __name__ == "__main__":
    main()