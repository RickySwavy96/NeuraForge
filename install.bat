@echo off
title NeuraForge - Complete Installation
color 0A

REM Change to script directory
cd /d "%~dp0"

echo.
echo ╔═══════════════════════════════════════════════════════╗
echo ║                                                       ║
echo ║            NeuraForge Complete Installation          ║
echo ║              AMD GPU Support via ZLUDA               ║
echo ║                                                       ║
echo ╚═══════════════════════════════════════════════════════╝
echo.
echo This script will:
echo   1. Check system prerequisites
echo   2. Create virtual environment
echo   3. Download and setup ZLUDA
echo   4. Install all dependencies
echo   5. Create launcher scripts
echo.
echo Installation directory: %CD%
echo.
pause

REM Step 1: Prerequisites check
echo.
echo ══════════════════════════════════════════════════════
echo   Step 1: Checking Prerequisites
echo ══════════════════════════════════════════════════════
echo.

set PREREQ_OK=1

REM Check Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Python not found!
    echo   Install Python 3.12.9 from https://www.python.org/downloads/
    set PREREQ_OK=0
) else (
    echo ✓ Python found
)

REM Check HIP SDK
if not defined HIP_PATH (
    echo ✗ HIP SDK not found!
    echo   Install AMD HIP SDK 6.4 from AMD website
    set PREREQ_OK=0
) else (
    echo ✓ HIP SDK found at: %HIP_PATH%
)

REM Check Git
git --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Git not found!
    echo   Install Git from https://git-scm.com/download/win
    set PREREQ_OK=0
) else (
    echo ✓ Git found
)

if %PREREQ_OK% EQU 0 (
    echo.
    echo ✗ Prerequisites check failed!
    echo   Please install missing software and run again.
    echo.
    pause
    exit /b 1
)

echo.
echo ✓ All prerequisites found!
echo.
pause

REM Step 2: Create virtual environment
echo.
echo ══════════════════════════════════════════════════════
echo   Step 2: Creating Virtual Environment
echo ══════════════════════════════════════════════════════
echo.

if exist "venv" (
    echo Virtual environment already exists.
    echo Skipping creation...
) else (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo ✗ Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created
)

REM Activate virtual environment
call venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Failed to activate virtual environment!
    pause
    exit /b 1
)

echo ✓ Virtual environment activated
echo.

REM Step 3: Upgrade pip
echo.
echo ══════════════════════════════════════════════════════
echo   Step 3: Upgrading pip
echo ══════════════════════════════════════════════════════
echo.

python -m pip install --upgrade pip
echo ✓ pip upgraded
echo.

REM Step 4: Download ZLUDA
echo.
echo ══════════════════════════════════════════════════════
echo   Step 4: Setting up ZLUDA for AMD GPU
echo ══════════════════════════════════════════════════════
echo.

if exist "zluda" (
    echo ZLUDA folder already exists.
    echo Skipping download...
) else (
    echo Downloading ZLUDA 3.9.5...
    echo This may take a few minutes...
    
    curl -L -o zluda.zip "https://github.com/lshqqytiger/ZLUDA/releases/download/rel.5e717459179dc272b7d7d23391f0fad66c7459cf/ZLUDA-windows-rocm6-amd64.zip"
    
    if %ERRORLEVEL% NEQ 0 (
        echo ✗ Failed to download ZLUDA!
        echo   Please check your internet connection.
        pause
        exit /b 1
    )
    
    echo Extracting ZLUDA...
    tar -xf zluda.zip
    if exist "ZLUDA-windows-rocm6-amd64" (
        move "ZLUDA-windows-rocm6-amd64" zluda
    )
    del zluda.zip
    
    echo ✓ ZLUDA installed
)

echo.

REM Step 5: Install PyTorch and dependencies
echo.
echo ══════════════════════════════════════════════════════
echo   Step 5: Installing Dependencies
echo ══════════════════════════════════════════════════════
echo.
echo This will take 10-15 minutes...
echo.

REM Install PyTorch first
echo Installing PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if %ERRORLEVEL% NEQ 0 (
    echo ✗ Failed to install PyTorch!
    pause
    exit /b 1
)

echo ✓ PyTorch installed
echo.

REM Install other dependencies
echo Installing other dependencies...
pip install transformers diffusers accelerate safetensors
pip install customtkinter pillow numpy
pip install sentencepiece protobuf einops opencv-python
pip install omegaconf pyyaml requests tqdm huggingface-hub

echo ✓ All dependencies installed
echo.

REM Step 6: Patch PyTorch with ZLUDA
echo.
echo ══════════════════════════════════════════════════════
echo   Step 6: Patching PyTorch with ZLUDA
echo ══════════════════════════════════════════════════════
echo.

set TORCH_LIB=venv\Lib\site-packages\torch\lib

if not exist "%TORCH_LIB%" (
    echo ✗ PyTorch lib folder not found!
    pause
    exit /b 1
)

echo Copying ZLUDA DLLs to PyTorch...
copy /Y "zluda\cublas.dll" "%TORCH_LIB%\cublas64_11.dll"
copy /Y "zluda\cusparse.dll" "%TORCH_LIB%\cusparse64_11.dll"
copy /Y "zluda\nvrtc.dll" "%TORCH_LIB%\nvrtc64_112_0.dll"

echo ✓ PyTorch patched with ZLUDA
echo.

REM Step 7: Create directory structure
echo.
echo ══════════════════════════════════════════════════════
echo   Step 7: Creating Directory Structure
echo ══════════════════════════════════════════════════════
echo.

mkdir models\text-generation 2>nul
mkdir models\image-generation 2>nul
mkdir models\vae 2>nul
mkdir models\lora 2>nul
mkdir outputs\text 2>nul
mkdir outputs\images 2>nul
mkdir cache\miopen 2>nul
mkdir cache\triton 2>nul

echo ✓ Directory structure created
echo.

REM Step 8: Create launcher
echo.
echo ══════════════════════════════════════════════════════
echo   Step 8: Creating Launcher
echo ══════════════════════════════════════════════════════
echo.

(
echo @echo off
echo title NeuraForge - AI Model Interface
echo.
echo REM Set environment variables for AMD GPU
echo set HIP_VISIBLE_DEVICES=0
echo set HSA_OVERRIDE_GFX_VERSION=10.3.0
echo set PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:128
echo set MIOPEN_DISABLE_CACHE=0
echo set MIOPEN_CUSTOM_CACHE_DIR=%%~dp0cache\miopen
echo set TRITON_CACHE_DIR=%%~dp0cache\triton
echo.
echo REM Activate virtual environment
echo call "%%~dp0venv\Scripts\activate.bat"
echo.
echo REM Run NeuraForge
echo python neuraforge_app.py
echo.
echo pause
) > launch_neuraforge.bat

echo ✓ Launcher created
echo.

REM Final summary
echo.
echo ╔═══════════════════════════════════════════════════════╗
echo ║                                                       ║
echo ║            Installation Complete! 🎉                 ║
echo ║                                                       ║
echo ╚═══════════════════════════════════════════════════════╝
echo.
echo NeuraForge has been successfully installed!
echo.
echo NEXT STEPS:
echo ═══════════════
echo.
echo 1. Download AI models:
echo    - Text models → models\text-generation\
echo    - Image models → models\image-generation\
echo.
echo 2. Run NeuraForge:
echo    Double-click: launch_neuraforge.bat
echo.
echo 3. First generation will be slower (ZLUDA compilation)
echo    Subsequent generations will be much faster!
echo.
echo IMPORTANT NOTES:
echo ═══════════════
echo • Ensure AMD drivers 25.5.1+ are installed
echo • First load of each model takes 1-2 minutes
echo • First generation adds 30-60 seconds for compilation
echo • Keep cache folders for better performance
echo.
echo For model downloads and usage guide, see:
echo   README.md and QUICKSTART.md
echo.
echo Happy Forging! 🔥
echo.
pause
