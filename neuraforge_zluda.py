"""
NeuraForge ZLUDA Integration Module
Handles AMD GPU acceleration via ZLUDA
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).parent
ZLUDA_PATH = PROJECT_ROOT / "zluda"

class ZLUDAManager:
    """Manages ZLUDA integration for AMD GPUs"""
    
    def __init__(self):
        self.zluda_available = self._check_zluda()
        self.hip_available = self._check_hip()
        
    def _check_zluda(self) -> bool:
        """Check if ZLUDA is properly installed"""
        zluda_exe = ZLUDA_PATH / "zluda.exe"
        if not zluda_exe.exists():
            print("⚠ ZLUDA not found. AMD GPU acceleration may not work optimally.")
            return False
        
        print(f"✓ ZLUDA found at: {ZLUDA_PATH}")
        return True
    
    def _check_hip(self) -> bool:
        """Check if AMD HIP SDK is installed"""
        hip_path = os.environ.get('HIP_PATH')
        
        if not hip_path:
            print("⚠ HIP_PATH not set. Please install AMD HIP SDK 6.4")
            return False
        
        hip_path_obj = Path(hip_path)
        if not hip_path_obj.exists():
            print(f"⚠ HIP SDK path not found: {hip_path}")
            return False
        
        print(f"✓ HIP SDK found at: {hip_path}")
        return True
    
    def setup_environment(self):
        """Setup environment variables for optimal AMD GPU performance"""
        
        # Core ZLUDA environment variables
        env_vars = {
            # HIP configuration
            'HIP_VISIBLE_DEVICES': '0',
            'HSA_OVERRIDE_GFX_VERSION': '10.3.0',  # Works for most AMD GPUs
            'PYTORCH_HIP_ALLOC_CONF': 'garbage_collection_threshold:0.8,max_split_size_mb:128',
            
            # Memory management
            'HIP_FORCE_DEV_KERNARG': '1',
            'ROCR_VISIBLE_DEVICES': '0',
            
            # Performance optimizations
            'MIOPEN_DISABLE_CACHE': '0',
            'MIOPEN_CUSTOM_CACHE_DIR': str(PROJECT_ROOT / "cache" / "miopen"),
            
            # ZLUDA specific
            'ZLUDA_ALLOW_UNSAFE': '1',
            
            # Triton cache
            'TRITON_CACHE_DIR': str(PROJECT_ROOT / "cache" / "triton"),
        }
        
        # Set environment variables
        for key, value in env_vars.items():
            os.environ[key] = value
            print(f"  {key} = {value}")
        
        # Add ZLUDA to PATH if available
        if self.zluda_available:
            zluda_bin = str(ZLUDA_PATH)
            current_path = os.environ.get('PATH', '')
            if zluda_bin not in current_path:
                os.environ['PATH'] = f"{zluda_bin};{current_path}"
                print(f"  Added ZLUDA to PATH")
        
        # Create cache directories
        (PROJECT_ROOT / "cache" / "miopen").mkdir(parents=True, exist_ok=True)
        (PROJECT_ROOT / "cache" / "triton").mkdir(parents=True, exist_ok=True)
        
        print("✓ Environment configured for AMD GPU")
    
    def get_gpu_info(self) -> Optional[dict]:
        """Get AMD GPU information"""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return None
            
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            
            return {
                'name': props.name,
                'total_memory': props.total_memory / (1024**3),  # Convert to GB
                'compute_capability': f"{props.major}.{props.minor}",
                'multi_processor_count': props.multi_processor_count,
            }
            
        except Exception as e:
            print(f"Error getting GPU info: {e}")
            return None
    
    def optimize_model_loading(self, model_type: str = "text") -> dict:
        """Get optimized loading parameters based on model type"""
        
        gpu_info = self.get_gpu_info()
        
        if not gpu_info:
            # CPU fallback
            return {
                'device_map': None,
                'torch_dtype': 'float32',
                'low_cpu_mem_usage': True,
            }
        
        vram_gb = gpu_info['total_memory']
        
        # Optimize based on VRAM
        if vram_gb >= 16:
            # High VRAM (16GB+) - Full precision possible
            return {
                'device_map': 'auto',
                'torch_dtype': 'float16',
                'low_cpu_mem_usage': True,
                'max_memory': {0: f"{int(vram_gb * 0.9)}GB"},
            }
        elif vram_gb >= 8:
            # Medium VRAM (8-16GB) - Use float16
            return {
                'device_map': 'auto',
                'torch_dtype': 'float16',
                'low_cpu_mem_usage': True,
                'max_memory': {0: f"{int(vram_gb * 0.85)}GB"},
                'load_in_8bit': False,
            }
        else:
            # Low VRAM (<8GB) - Aggressive optimization
            return {
                'device_map': 'auto',
                'torch_dtype': 'float16',
                'low_cpu_mem_usage': True,
                'max_memory': {0: f"{int(vram_gb * 0.8)}GB"},
                'load_in_8bit': True,  # Requires bitsandbytes
            }
    
    def clear_cache(self):
        """Clear GPU cache"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("✓ GPU cache cleared")
        except Exception as e:
            print(f"Error clearing cache: {e}")
    
    def benchmark_gpu(self) -> Tuple[bool, float]:
        """Quick GPU benchmark to verify ZLUDA is working"""
        try:
            import torch
            import time
            
            if not torch.cuda.is_available():
                return False, 0.0
            
            device = torch.device('cuda')
            
            # Simple matrix multiplication benchmark
            size = 2048
            a = torch.randn(size, size, device=device, dtype=torch.float16)
            b = torch.randn(size, size, device=device, dtype=torch.float16)
            
            # Warmup
            for _ in range(3):
                _ = torch.matmul(a, b)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.time()
            for _ in range(10):
                _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            ops = (2 * size ** 3 * 10) / elapsed
            tflops = ops / 1e12
            
            print(f"✓ GPU Benchmark: {tflops:.2f} TFLOPS")
            
            return True, tflops
            
        except Exception as e:
            print(f"✗ GPU benchmark failed: {e}")
            return False, 0.0
    
    def troubleshoot(self):
        """Run diagnostics and provide troubleshooting info"""
        print("\n" + "="*60)
        print("  ZLUDA / AMD GPU Diagnostics")
        print("="*60 + "\n")
        
        # Check ZLUDA
        print("1. ZLUDA Installation:")
        if self.zluda_available:
            print(f"   ✓ ZLUDA found at: {ZLUDA_PATH}")
        else:
            print(f"   ✗ ZLUDA not found")
            print(f"   → Run setup script to install ZLUDA")
        
        # Check HIP SDK
        print("\n2. AMD HIP SDK:")
        if self.hip_available:
            hip_path = os.environ.get('HIP_PATH')
            print(f"   ✓ HIP SDK found at: {hip_path}")
        else:
            print("   ✗ HIP SDK not found")
            print("   → Install AMD HIP SDK 6.4 from:")
            print("      https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html")
        
        # Check PyTorch CUDA
        print("\n3. PyTorch CUDA Support:")
        try:
            import torch
            if torch.cuda.is_available():
                print(f"   ✓ CUDA available")
                print(f"   Device: {torch.cuda.get_device_name(0)}")
            else:
                print("   ✗ CUDA not available")
                print("   → Check ZLUDA installation and PyTorch patching")
        except ImportError:
            print("   ✗ PyTorch not installed")
        
        # Check GPU info
        print("\n4. GPU Information:")
        gpu_info = self.get_gpu_info()
        if gpu_info:
            print(f"   Name: {gpu_info['name']}")
            print(f"   VRAM: {gpu_info['total_memory']:.1f} GB")
            print(f"   Compute: {gpu_info['compute_capability']}")
        else:
            print("   ✗ Cannot get GPU info")
        
        # Benchmark
        print("\n5. GPU Performance Test:")
        success, tflops = self.benchmark_gpu()
        if not success:
            print("   ✗ Benchmark failed")
            print("   → Check ZLUDA setup and driver installation")
        
        # Environment variables
        print("\n6. Environment Variables:")
        important_vars = [
            'HIP_PATH', 'HIP_VISIBLE_DEVICES', 'HSA_OVERRIDE_GFX_VERSION',
            'PYTORCH_HIP_ALLOC_CONF'
        ]
        for var in important_vars:
            value = os.environ.get(var, 'NOT SET')
            symbol = '✓' if value != 'NOT SET' else '✗'
            print(f"   {symbol} {var}: {value}")
        
        print("\n" + "="*60 + "\n")
        
        # Recommendations
        if not (self.zluda_available and self.hip_available):
            print("RECOMMENDATIONS:")
            print("1. Run the setup script: python neuraforge_setup.py")
            print("2. Ensure AMD GPU drivers 25.5.1+ are installed")
            print("3. Restart after installing HIP SDK")
            print()


def initialize_zluda():
    """Initialize ZLUDA system - call this at app startup"""
    print("\n" + "="*60)
    print("  Initializing AMD GPU Support (ZLUDA)")
    print("="*60 + "\n")
    
    zluda_mgr = ZLUDAManager()
    
    # Setup environment
    zluda_mgr.setup_environment()
    
    # Get GPU info
    gpu_info = zluda_mgr.get_gpu_info()
    if gpu_info:
        print(f"\n✓ GPU Detected: {gpu_info['name']}")
        print(f"  VRAM: {gpu_info['total_memory']:.1f} GB")
    else:
        print("\n⚠ No GPU detected or ZLUDA not working")
        print("  Running in CPU mode")
    
    print("\n" + "="*60 + "\n")
    
    return zluda_mgr


if __name__ == "__main__":
    # Run diagnostics
    zluda_mgr = ZLUDAManager()
    zluda_mgr.troubleshoot()
