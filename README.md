# ⚡ NeuraForge

**Unified AI Model Interface for Windows with AMD GPU Support**

NeuraForge is a powerful, user-friendly desktop application that allows you to run AI models locally on your Windows PC with full AMD GPU acceleration via ZLUDA. No cloud services, no subscriptions - just pure local AI power.

![Python 3.12.9](https://img.shields.io/badge/Python-3.12.9-blue)
![AMD GPU](https://img.shields.io/badge/AMD-GPU%20Accelerated-red)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- 🚀 **Unified Interface**: Run both text generation and image generation models in one application
- 💪 **AMD GPU Support**: Full ZLUDA integration for AMD Radeon GPUs (RX 6000/7000/9000 series)
- 📦 **Model Flexibility**: Support for popular models:
  - **Text**: Qwen, DeepSeek, Llama, Mistral, Yi, and more
  - **Image**: Stable Diffusion, SDXL, Flux, and custom models
- 🎨 **Modern UI**: Beautiful dark-themed interface built with CustomTkinter
- 🔄 **Virtual Environment**: Isolated Python environment for clean dependency management
- 💾 **Local Storage**: All models and outputs stored locally on your machine
- ⚡ **Optimized Performance**: Automatic memory management and model optimization

---

## 📋 Requirements

### Hardware Requirements
- **GPU**: AMD Radeon RX 6950 XT (or similar RX 6000/7000/9000 series)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB+ free space (models can be large)
- **OS**: Windows 10/11 (64-bit)

### Software Requirements
- **Python**: 3.12.9 (from [python.org](https://www.python.org/downloads/), NOT Microsoft Store)
- **AMD HIP SDK**: 6.4 ([Download here](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html))
- **AMD GPU Drivers**: 25.5.1 or newer
- **Git**: For cloning repository ([Download](https://git-scm.com/download/win))
- **Visual Studio Build Tools**: ([Download](https://aka.ms/vs/17/release/vs_BuildTools.exe))

---

## 🚀 Installation

### Step 1: Install Prerequisites

1. **Install Python 3.12.9**
   - Download from [python.org](https://www.python.org/downloads/windows/)
   - ⚠️ **IMPORTANT**: Check "Add Python to PATH" during installation
   - ⚠️ **DO NOT** install from Microsoft Store

2. **Install AMD HIP SDK 6.4**
   - Download "Windows 10 & 11 6.4.2 HIP SDK" from [AMD ROCm Hub](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)
   - Install with default settings
   - Verify environment variables are set:
     - `HIP_PATH` = `C:\Program Files\AMD\ROCm\6.4\`
     - `HIP_PATH_64` = `C:\Program Files\AMD\ROCm\6.4\`
   - Add to PATH: `C:\Program Files\AMD\ROCm\6.4\bin`

3. **Install Visual Studio Build Tools**
   - Download from [here](https://aka.ms/vs/17/release/vs_BuildTools.exe)
   - Select "Desktop development with C++"
   - Install and restart

4. **Install Git**
   - Download from [git-scm.com](https://git-scm.com/download/win)
   - Check "Use Git from Windows Command Prompt"

5. **Update AMD GPU Drivers**
   - Download latest drivers (25.5.1+) from AMD website
   - Install and restart

### Step 2: Install NeuraForge

1. **Open Command Prompt** (as Administrator recommended)

2. **Navigate to your desired installation location**:
   ```cmd
   cd E:\
   ```
   
   ⚠️ **DO NOT** install in:
   - Program Files
   - User directories with special characters
   - Windows system folders

3. **Run the setup script**:
   ```cmd
   python neuraforge_setup.py
   ```

4. **Wait for installation to complete** (10-20 minutes depending on internet speed)

### Step 3: Verify Installation

After installation completes, you should see:
```
✓ NeuraForge has been successfully installed!
```

Run diagnostics to verify everything is working:
```cmd
python neuraforge_zluda.py
```

This will check:
- ✓ ZLUDA installation
- ✓ HIP SDK detection
- ✓ GPU recognition
- ✓ PyTorch CUDA support
- ✓ Performance benchmark

---

## 📥 Downloading Models

### Where to Get Models

#### Text Generation Models
- **Hugging Face**: [huggingface.co/models](https://huggingface.co/models)
  - Qwen 2.5 (7B, 14B, 32B)
  - DeepSeek (7B, 33B)
  - Llama 3.1/3.2
  - Mistral (7B, Mixtral)

#### Image Generation Models
- **Hugging Face**: [huggingface.co/models](https://huggingface.co/models)
  - Stable Diffusion 1.5
  - Stable Diffusion XL
  - Flux.1 [schnell/dev]
- **CivitAI**: [civitai.com](https://civitai.com)
  - Community fine-tuned models
  - LoRA models

### Model Installation

1. **Download models** to:
   - Text models: `E:\NeuraForge\models\text-generation\`
   - Image models: `E:\NeuraForge\models\image-generation\`

2. **Supported formats**:
   - `.safetensors` (recommended)
   - `.ckpt`
   - `.bin`
   - Hugging Face folder structure

3. **Example structure**:
   ```
   E:\NeuraForge\
   ├── models\
   │   ├── text-generation\
   │   │   ├── Qwen2.5-7B\
   │   │   │   ├── model.safetensors
   │   │   │   ├── config.json
   │   │   │   └── tokenizer files...
   │   │   └── deepseek-llm-7b-base\
   │   └── image-generation\
   │       ├── stable-diffusion-xl-base-1.0\
   │       ├── flux-schnell\
   │       └── custom-model.safetensors
   ```

### Recommended Models for AMD RX 6950 XT

**Text Generation** (16GB VRAM):
- ✅ Qwen 2.5 7B (recommended)
- ✅ DeepSeek 7B
- ✅ Llama 3.1 8B
- ✅ Mistral 7B
- ⚠️ 13B+ models (may need optimization)

**Image Generation**:
- ✅ Stable Diffusion 1.5 (512x512)
- ✅ Stable Diffusion XL (1024x1024)
- ✅ Flux.1 Schnell (1024x1024)
- ✅ Most community models

---

## 🎮 Usage

### Starting NeuraForge

**Option 1: Use Launcher**
```cmd
launch_neuraforge.bat
```

**Option 2: Manual Start**
```cmd
cd E:\NeuraForge
venv\Scripts\activate
python neuraforge_app.py
```

### Text Generation

1. Click **"📝 Text Generation"** in sidebar
2. Click **"♻️ Refresh Models"** to scan for models
3. Select a model from dropdown
4. Click **"🔄 Load Selected Model"** (first load takes 1-2 minutes)
5. Enter your prompt
6. Adjust parameters:
   - **Max Length**: 50-1000 tokens
   - **Temperature**: 0.1 (focused) to 2.0 (creative)
7. Click **"🚀 Generate Text"**
8. Output saved to `outputs/text/`

### Image Generation

1. Click **"🎨 Image Generation"** in sidebar
2. Select and load an image model
3. Enter your prompt (describe what you want)
4. Enter negative prompt (what to avoid)
5. Adjust parameters:
   - **Steps**: 20-50 (more = better quality, slower)
   - **Guidance**: 7-12 (how closely to follow prompt)
   - **Width/Height**: 512, 768, or 1024
6. Click **"🎨 Generate Image"**
7. Image saved to `outputs/images/`

### Tips for Best Performance

- **First generation is slow**: ZLUDA needs to compile kernels (one-time)
- **Close other GPU-intensive apps**: Free up VRAM
- **Start with smaller models**: Test before downloading large models
- **Monitor VRAM usage**: Windows Task Manager → Performance → GPU
- **Use batch generation**: Generate multiple images without reloading

---

## 🔧 Configuration

### Environment Variables (Auto-configured)

The setup script configures these automatically:
```
HIP_VISIBLE_DEVICES=0
HSA_OVERRIDE_GFX_VERSION=10.3.0
PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:128
```

### Manual Optimization (Optional)

For specific AMD GPUs, you may need to adjust `HSA_OVERRIDE_GFX_VERSION`:

| GPU Series | Override Value |
|------------|----------------|
| RX 6900 XT, 6950 XT | 10.3.0 |
| RX 6800, 6800 XT | 10.3.0 |
| RX 6700 XT | 10.3.0 |
| RX 7900 XTX, 7900 XT | 11.0.0 |
| RX 7800 XT, 7700 XT | 11.0.0 |

Edit `launch_neuraforge.bat` to change this value.

---

## 🐛 Troubleshooting

### GPU Not Detected

**Problem**: "No GPU detected, using CPU"

**Solutions**:
1. Run diagnostics: `python neuraforge_zluda.py`
2. Check AMD drivers are 25.5.1+
3. Verify HIP SDK environment variables
4. Restart computer after driver/SDK installation
5. Check Windows Device Manager for GPU

### Model Loading Errors

**Problem**: "Failed to load model"

**Solutions**:
1. Verify model files are complete (not corrupted)
2. Check VRAM availability (close other apps)
3. Try smaller model first
4. Check model format is supported
5. Look for error details in console

### Out of Memory Errors

**Problem**: "CUDA out of memory"

**Solutions**:
1. Close other GPU applications
2. Use smaller model
3. Reduce batch size (for image generation)
4. Lower resolution (for images)
5. Clear cache: Delete `cache/` folder and restart

### Slow Generation

**Problem**: Generation is very slow

**Solutions**:
1. First generation is always slow (ZLUDA compilation)
2. Subsequent generations should be faster
3. Check GPU usage in Task Manager
4. Ensure not running on CPU (check console output)
5. Try clearing cache folders:
   - `C:\Users\[username]\AppData\Local\ZLUDA\ComputeCache`
   - `C:\Users\[username]\.miopen`
   - `C:\Users\[username]\.triton`

### Black Images (Image Generation)

**Problem**: Generated images are black

**Solutions**:
1. Check model is compatible with ZLUDA
2. Try different VAE (if using Flux)
3. Reduce guidance scale
4. Increase inference steps
5. Try SD 1.5 first (most compatible)

### Import Errors

**Problem**: "ModuleNotFoundError" or import errors

**Solutions**:
1. Activate virtual environment: `venv\Scripts\activate`
2. Reinstall dependencies: `pip install -r requirements.txt`
3. Check Python version: `python --version` (should be 3.12.9)
4. Verify using python.org version, not Microsoft Store

---

## 📊 Performance Expectations

### AMD RX 6950 XT (16GB VRAM)

**Text Generation**:
- Qwen 2.5 7B: ~30-40 tokens/second
- Llama 3.1 8B: ~25-35 tokens/second
- Load time: 30-60 seconds (first time longer)

**Image Generation**:
- SD 1.5 (512x512, 30 steps): ~15-20 seconds
- SDXL (1024x1024, 30 steps): ~40-60 seconds
- Flux Schnell (1024x1024, 4 steps): ~10-15 seconds

*Note: First generation adds 30-60 seconds for ZLUDA compilation*

---

## 🔄 Updating

### Update NeuraForge

```cmd
cd E:\NeuraForge
git pull
python neuraforge_setup.py
```

### Update Dependencies

```cmd
cd E:\NeuraForge
venv\Scripts\activate
pip install --upgrade torch transformers diffusers
```

---

## 📁 Project Structure

```
E:\NeuraForge\
├── neuraforge_setup.py      # Setup and installation script
├── neuraforge_app.py         # Main application
├── neuraforge_zluda.py       # ZLUDA integration module
├── launch_neuraforge.bat     # Launcher script
├── config.json               # User configuration
├── venv\                     # Virtual environment
├── zluda\                    # ZLUDA binaries
├── models\
│   ├── text-generation\      # Place text models here
│   ├── image-generation\     # Place image models here
│   ├── vae\                  # VAE models
│   └── lora\                 # LoRA models
├── outputs\
│   ├── text\                 # Generated text outputs
│   └── images\               # Generated images
└── cache\                    # Compilation cache
    ├── miopen\
    └── triton\
```

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📜 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- **ZLUDA**: [lshqqytiger/ZLUDA](https://github.com/lshqqytiger/ZLUDA) - CUDA on AMD GPUs
- **ComfyUI-Zluda**: [patientx/ComfyUI-Zluda](https://github.com/patientx/ComfyUI-Zluda) - Reference implementation
- **Hugging Face**: Transformers and Diffusers libraries
- **CustomTkinter**: Modern UI framework

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/NeuraForge/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/NeuraForge/discussions)

---

## ⚠️ Important Notes

- **First Launch**: Initial model loading and generation will be slower due to ZLUDA kernel compilation
- **Cache Management**: ZLUDA builds cache files - keep them for better performance
- **Driver Updates**: AMD driver updates may require cache clearing
- **Model Compatibility**: Not all models work perfectly with ZLUDA - test before downloading large models
- **VRAM Monitoring**: Keep an eye on VRAM usage to avoid OOM errors

---

**Built with ❤️ for the AMD GPU community**

Happy Forging! 🔥
