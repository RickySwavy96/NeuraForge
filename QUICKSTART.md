# ⚡ NeuraForge Quick Start Guide

**Get up and running in 30 minutes!**

---

## 📦 What You Need

Before starting, make sure you have:

- ✅ Windows 10/11 (64-bit)
- ✅ AMD Radeon RX 6000/7000/9000 series GPU (16GB+ VRAM recommended)
- ✅ 50GB+ free disk space
- ✅ Good internet connection (for downloads)

---

## 🚀 Installation Steps

### 1. Install Prerequisites (15 minutes)

Download and install in this order:

#### A. Python 3.12.9
- Go to: https://www.python.org/downloads/windows/
- Download "Windows installer (64-bit)"
- **Important:** Check "Add Python to PATH" during install
- **Do NOT** use Microsoft Store version

#### B. AMD HIP SDK 6.4
- Go to: https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html
- Download "Windows 10 & 11 6.4.2 HIP SDK"
- Install with default settings
- Restart after installation

#### C. Visual Studio Build Tools
- Go to: https://aka.ms/vs/17/release/vs_BuildTools.exe
- Select "Desktop development with C++"
- Install and continue (no restart needed yet)

#### D. Git
- Go to: https://git-scm.com/download/win
- Install with defaults
- Make sure "Use Git from Windows Command Prompt" is checked

#### E. Update AMD Drivers
- Download latest drivers from AMD (version 25.5.1+)
- Install and **restart your PC**

---

### 2. Install NeuraForge (10 minutes)

After restarting, open Command Prompt **as Administrator**:

```cmd
# Navigate to where you want to install (NOT Program Files!)
cd E:\

# Download setup script
# Copy neuraforge_setup.py to E:\NeuraForge\

# Run setup
cd NeuraForge
python neuraforge_setup.py
```

Wait for installation to complete. You'll see:
```
✓ NeuraForge has been successfully installed!
```

---

### 3. Download Your First Model (5+ minutes)

#### Option A: Text Generation (Recommended for first test)

Go to Hugging Face and download a small model:

**Qwen 2.5 7B** (Recommended):
1. Visit: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
2. Click "Files and versions"
3. Download these files to `E:\NeuraForge\models\text-generation\Qwen2.5-7B-Instruct\`:
   - `config.json`
   - `model.safetensors` (or all shard files if split)
   - `tokenizer.json`
   - `tokenizer_config.json`
   - `special_tokens_map.json`

Or use Git LFS (faster):
```cmd
cd E:\NeuraForge\models\text-generation
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
```

#### Option B: Image Generation

**Stable Diffusion 1.5** (Easiest):
```cmd
cd E:\NeuraForge\models\image-generation
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```

---

### 4. Run NeuraForge!

Double-click: `launch_neuraforge.bat`

Or from Command Prompt:
```cmd
cd E:\NeuraForge
launch_neuraforge.bat
```

---

## 🎮 Your First Generation

### Text Generation Test

1. App opens → Click **"📝 Text Generation"**
2. Click **"♻️ Refresh Models"**
3. Select your downloaded model
4. Click **"🔄 Load Selected Model"** (wait 1-2 minutes)
5. Type a prompt: `"Write a short story about a robot"`
6. Click **"🚀 Generate Text"**

**First generation takes 30-60 seconds extra** (ZLUDA compilation)
Second generation will be much faster!

### Image Generation Test

1. Click **"🎨 Image Generation"**
2. Load your model (same process)
3. Prompt: `"a beautiful sunset over mountains, highly detailed"`
4. Negative: `"blurry, low quality"`
5. Click **"🎨 Generate Image"**

---

## ⚡ Performance Tips

### First Time Setup
- **First load**: 1-2 minutes (model loading)
- **First generation**: +30-60 seconds (ZLUDA compilation)
- **Subsequent**: Much faster!

### Optimal Settings

**For AMD RX 6950 XT (16GB VRAM):**

**Text Generation:**
- Max Length: 200-500 tokens
- Temperature: 0.7 (balanced)

**Image Generation:**
- Resolution: 512x512 (fast) or 1024x1024 (quality)
- Steps: 20-30 (balanced)
- Guidance: 7-9 (balanced)

---

## 🐛 Quick Fixes

### "No GPU detected"
```cmd
# Run diagnostics
python neuraforge_zluda.py
```
- Check AMD drivers installed
- Check HIP SDK installed
- Verify HIP_PATH environment variable set
- Restart PC

### "Out of memory"
- Close other GPU applications
- Use smaller model
- Lower image resolution
- Reduce max length for text

### "Model loading failed"
- Check model files are complete
- Try different model
- Check VRAM available
- Look at console for error details

### "Generation is very slow"
- First generation is always slow (cache building)
- Second generation should be faster
- Check GPU usage in Task Manager
- Clear cache: run `cache-clean.bat`

---

## 📂 Where Everything Is

```
E:\NeuraForge\
├── launch_neuraforge.bat    ← Double-click to start
├── cache-clean.bat           ← Run if having issues
├── models\
│   ├── text-generation\      ← Put text models here
│   └── image-generation\     ← Put image models here
└── outputs\
    ├── text\                 ← Generated text saved here
    └── images\               ← Generated images saved here
```

---

## 🎯 Next Steps

Once everything works:

1. **Download more models** from Hugging Face
2. **Experiment with settings** - find what works best
3. **Check outputs folder** - all generations are saved
4. **Read README.md** - for advanced features

---

## 📞 Need Help?

**Can't get it working?**

1. Run: `python neuraforge_zluda.py` - shows diagnostics
2. Check the detailed README.md
3. Look at console output for errors
4. Verify all prerequisites installed

**Common Issues:**
- Forgot to restart after HIP SDK install
- Using Python from Microsoft Store (wrong!)
- AMD drivers too old
- Not running as Administrator

---

## ✅ Checklist

Before reporting issues, verify:

- [ ] Python 3.12.9 from python.org (not Store)
- [ ] AMD HIP SDK 6.4 installed
- [ ] AMD GPU drivers 25.5.1+ installed
- [ ] Visual Studio Build Tools installed
- [ ] PC restarted after installations
- [ ] HIP_PATH environment variable exists
- [ ] Model files completely downloaded
- [ ] Setup script completed successfully
- [ ] Running from correct directory

---

**Ready to forge? Let's go! 🔥**

If you encounter issues, run diagnostics first:
```cmd
python neuraforge_zluda.py
```

This will tell you exactly what's wrong!
