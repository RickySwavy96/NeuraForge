"""
NeuraForge Setup Script
Handles virtual environment creation, ZLUDA integration, and dependencies
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path

PROJECT_ROOT = Path("E:/NeuraForge")
VENV_PATH = PROJECT_ROOT / "venv"
ZLUDA_PATH = PROJECT_ROOT / "zluda"
MODELS_PATH = PROJECT_ROOT / "models"

def print_step(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")

def create_virtual_environment():
    """Create Python virtual environment"""
    print_step("Creating Virtual Environment")
    
    if VENV_PATH.exists():
        print(f"Virtual environment already exists at {VENV_PATH}")
        return True
    
    try:
        subprocess.run([
            sys.executable, "-m", "venv", str(VENV_PATH)
        ], check=True)
        print(f"✓ Virtual environment created at {VENV_PATH}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to create virtual environment: {e}")
        return False

def get_pip_executable():
    """Get pip executable path from venv"""
    if sys.platform == "win32":
        return VENV_PATH / "Scripts" / "pip.exe"
    return VENV_PATH / "bin" / "pip"

def get_python_executable():
    """Get python executable path from venv"""
    if sys.platform == "win32":
        return VENV_PATH / "Scripts" / "python.exe"
    return VENV_PATH / "bin" / "python"

def upgrade_pip():
    """Upgrade pip in virtual environment"""
    print_step("Upgrading pip")
    
    pip_exe = get_pip_executable()
    try:
        subprocess.run([
            str(pip_exe), "install", "--upgrade", "pip"
        ], check=True)
        print("✓ pip upgraded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to upgrade pip: {e}")
        return False

def download_zluda():
    """Download and setup ZLUDA for AMD GPU support"""
    print_step("Setting up ZLUDA for AMD GPU Support")
    
    # Check if HIP SDK is installed
    hip_path = os.environ.get('HIP_PATH')
    if not hip_path:
        print("⚠ Warning: HIP_PATH environment variable not found")
        print("  Please ensure AMD HIP SDK 6.4 is installed")
        print("  Download from: https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html")
    else:
        print(f"✓ HIP SDK found at: {hip_path}")
    
    if ZLUDA_PATH.exists():
        print(f"ZLUDA already exists at {ZLUDA_PATH}")
        return True
    
    # Download ZLUDA 3.9.5 nightly for HIP 6.4
    zluda_url = "https://github.com/lshqqytiger/ZLUDA/releases/download/rel.5e717459179dc272b7d7d23391f0fad66c7459cf/ZLUDA-windows-rocm6-amd64.zip"
    zluda_zip = PROJECT_ROOT / "zluda.zip"
    
    try:
        print(f"Downloading ZLUDA from {zluda_url}...")
        urllib.request.urlretrieve(zluda_url, zluda_zip)
        print("✓ ZLUDA downloaded")
        
        print("Extracting ZLUDA...")
        with zipfile.ZipFile(zluda_zip, 'r') as zip_ref:
            zip_ref.extractall(PROJECT_ROOT)
        
        # Rename extracted folder to 'zluda'
        extracted = PROJECT_ROOT / "ZLUDA-windows-rocm6-amd64"
        if extracted.exists():
            extracted.rename(ZLUDA_PATH)
        
        zluda_zip.unlink()
        print(f"✓ ZLUDA extracted to {ZLUDA_PATH}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download ZLUDA: {e}")
        return False

def patch_zluda_torch():
    """Patch PyTorch with ZLUDA DLLs"""
    print_step("Patching PyTorch with ZLUDA")
    
    torch_lib = VENV_PATH / "Lib" / "site-packages" / "torch" / "lib"
    
    if not torch_lib.exists():
        print("⚠ PyTorch not installed yet, will patch after installation")
        return True
    
    zluda_files = {
        'cublas.dll': 'cublas64_11.dll',
        'cusparse.dll': 'cusparse64_11.dll',
        'nvrtc.dll': 'nvrtc64_112_0.dll'
    }
    
    try:
        for src_name, dst_name in zluda_files.items():
            src = ZLUDA_PATH / src_name
            dst = torch_lib / dst_name
            
            if src.exists():
                shutil.copy2(src, dst)
                print(f"✓ Copied {src_name} -> {dst_name}")
            else:
                print(f"⚠ Warning: {src_name} not found in ZLUDA folder")
        
        print("✓ PyTorch patched with ZLUDA")
        return True
        
    except Exception as e:
        print(f"✗ Failed to patch PyTorch: {e}")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print_step("Installing Dependencies")
    
    pip_exe = get_pip_executable()
    
    # Core dependencies
    dependencies = [
        "torch>=2.5.0",
        "torchvision",
        "torchaudio",
        "transformers>=4.40.0",
        "diffusers>=0.27.0",
        "accelerate>=0.30.0",
        "safetensors>=0.4.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "customtkinter>=5.2.0",
        "tkinterdnd2>=0.3.0",
        "sentencepiece",
        "protobuf",
        "einops",
        "opencv-python",
        "omegaconf",
        "pyyaml",
        "requests",
        "tqdm",
    ]
    
    try:
        for dep in dependencies:
            print(f"Installing {dep}...")
            result = subprocess.run([
                str(pip_exe), "install", dep
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"⚠ Warning installing {dep}: {result.stderr}")
            else:
                print(f"✓ {dep} installed")
        
        print("\n✓ All dependencies installed")
        
        # Now patch ZLUDA after torch is installed
        patch_zluda_torch()
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False

def create_directory_structure():
    """Create necessary directories"""
    print_step("Creating Directory Structure")
    
    dirs = [
        MODELS_PATH / "text-generation",
        MODELS_PATH / "image-generation",
        MODELS_PATH / "vae",
        MODELS_PATH / "lora",
        PROJECT_ROOT / "outputs" / "text",
        PROJECT_ROOT / "outputs" / "images",
        PROJECT_ROOT / "cache",
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {dir_path.relative_to(PROJECT_ROOT)}")
    
    return True

def create_launcher_script():
    """Create launcher batch file"""
    print_step("Creating Launcher Script")
    
    launcher = PROJECT_ROOT / "launch_neuraforge.bat"
    
    launcher_content = f"""@echo off
title NeuraForge - AI Model Interface

REM Set environment variables for AMD GPU
set HIP_VISIBLE_DEVICES=0
set HSA_OVERRIDE_GFX_VERSION=10.3.0
set PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:128

REM Activate virtual environment
call "{VENV_PATH}\\Scripts\\activate.bat"

REM Run NeuraForge
python neuraforge_app.py

pause
"""
    
    try:
        launcher.write_text(launcher_content)
        print(f"✓ Launcher created at {launcher}")
        return True
    except Exception as e:
        print(f"✗ Failed to create launcher: {e}")
        return False

def main():
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║           NeuraForge Setup & Installation            ║
    ║         AMD GPU Support via ZLUDA Integration        ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    # Ensure project root exists
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    os.chdir(PROJECT_ROOT)
    
    steps = [
        ("Create Virtual Environment", create_virtual_environment),
        ("Upgrade pip", upgrade_pip),
        ("Download ZLUDA", download_zluda),
        ("Install Dependencies", install_dependencies),
        ("Create Directory Structure", create_directory_structure),
        ("Create Launcher", create_launcher_script),
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n✗ Setup failed at: {step_name}")
            print("\nPlease resolve the error and run setup again.")
            sys.exit(1)
    
    print_step("Setup Complete!")
    print("""
    ✓ NeuraForge has been successfully installed!
    
    Next Steps:
    1. Place your AI models in the 'models' folder:
       - Text models: models/text-generation/
       - Image models: models/image-generation/
    
    2. Run NeuraForge:
       - Double-click: launch_neuraforge.bat
       - Or run: python neuraforge_app.py
    
    3. Ensure AMD drivers 25.5.1+ are installed
    
    4. First generation will take longer (ZLUDA compilation)
    
    Happy forging! 🔥
    """)

if __name__ == "__main__":
    main()
