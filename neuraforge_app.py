"""
NeuraForge - Unified AI Model Interface
Text Generation & Image Generation with AMD GPU Support
"""

import os
import sys
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import customtkinter as ctk
from tkinter import filedialog, scrolledtext
from PIL import Image, ImageTk

# Import ZLUDA manager
try:
    from neuraforge_zluda import initialize_zluda
    ZLUDA_AVAILABLE = True
except ImportError:
    print("⚠ Warning: ZLUDA module not found. AMD GPU acceleration may not work.")
    ZLUDA_AVAILABLE = False

import torch

# Set appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

PROJECT_ROOT = Path(__file__).parent
MODELS_PATH = PROJECT_ROOT / "models"
OUTPUTS_PATH = PROJECT_ROOT / "outputs"
CONFIG_PATH = PROJECT_ROOT / "config.json"

class ModelManager:
    """Manages loading and caching of AI models"""
    
    def __init__(self):
        self.loaded_models = {}
        self.device = self._detect_device()
        
    def _detect_device(self) -> str:
        """Detect available compute device"""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU Detected: {device_name}")
            return "cuda"
        else:
            print("⚠ No GPU detected, using CPU")
            return "cpu"
    
    def get_available_models(self, model_type: str) -> List[Dict]:
        """Scan and return available models"""
        model_dir = MODELS_PATH / model_type
        if not model_dir.exists():
            return []
        
        models = []
        for item in model_dir.iterdir():
            if item.is_dir() or item.suffix in ['.safetensors', '.ckpt', '.bin', '.pth']:
                models.append({
                    'name': item.name,
                    'path': str(item),
                    'size': self._get_size_str(item)
                })
        return models
    
    def _get_size_str(self, path: Path) -> str:
        """Get human-readable file/folder size"""
        try:
            if path.is_file():
                size = path.stat().st_size
            else:
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size < 1024:
                    return f"{size:.1f} {unit}"
                size /= 1024
            return f"{size:.1f} PB"
        except:
            return "Unknown"
    
    def load_text_model(self, model_path: str, callback=None):
        """Load a text generation model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            if callback:
                callback("Loading tokenizer...")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            if callback:
                callback("Loading model...")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                model = model.to(self.device)
            
            self.loaded_models['text'] = {
                'model': model,
                'tokenizer': tokenizer,
                'path': model_path
            }
            
            if callback:
                callback("Model loaded successfully!")
            
            return True
            
        except Exception as e:
            if callback:
                callback(f"Error loading model: {str(e)}")
            return False
    
    def load_image_model(self, model_path: str, callback=None):
        """Load an image generation model"""
        try:
            from diffusers import StableDiffusionPipeline, FluxPipeline
            
            if callback:
                callback("Loading image generation model...")
            
            # Try to detect model type
            if 'flux' in model_path.lower():
                pipeline = FluxPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            else:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            
            pipeline = pipeline.to(self.device)
            
            self.loaded_models['image'] = {
                'pipeline': pipeline,
                'path': model_path
            }
            
            if callback:
                callback("Model loaded successfully!")
            
            return True
            
        except Exception as e:
            if callback:
                callback(f"Error loading model: {str(e)}")
            return False
    
    def generate_text(self, prompt: str, max_length: int = 200, temperature: float = 0.7):
        """Generate text using loaded model"""
        if 'text' not in self.loaded_models:
            return "Error: No text model loaded"
        
        try:
            model_data = self.loaded_models['text']
            tokenizer = model_data['tokenizer']
            model = model_data['model']
            
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
            
        except Exception as e:
            return f"Error generating text: {str(e)}"
    
    def generate_image(self, prompt: str, negative_prompt: str = "", 
                      steps: int = 30, guidance: float = 7.5,
                      width: int = 512, height: int = 512):
        """Generate image using loaded model"""
        if 'image' not in self.loaded_models:
            return None, "Error: No image model loaded"
        
        try:
            pipeline = self.loaded_models['image']['pipeline']
            
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height
            )
            
            return result.images[0], "Image generated successfully!"
            
        except Exception as e:
            return None, f"Error generating image: {str(e)}"


class NeuraForgeApp(ctk.CTk):
    """Main NeuraForge Application"""
    
    def __init__(self):
        super().__init__()
        
        self.title("NeuraForge - AI Model Interface")
        self.geometry("1400x900")
        
        self.model_manager = ModelManager()
        self.current_mode = "text"  # text or image
        
        self.load_config()
        self.create_ui()
        
    def load_config(self):
        """Load saved configuration"""
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, 'r') as f:
                    self.config = json.load(f)
            except:
                self.config = {}
        else:
            self.config = {}
    
    def save_config(self):
        """Save configuration"""
        try:
            with open(CONFIG_PATH, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def create_ui(self):
        """Create the main user interface"""
        
        # Configure grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Sidebar
        self.create_sidebar()
        
        # Main content area
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        
        # Create mode-specific interfaces
        self.create_text_interface()
        self.create_image_interface()
        
        # Show text interface by default
        self.switch_mode("text")
    
    def create_sidebar(self):
        """Create sidebar with navigation and model management"""
        
        sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        sidebar.grid_rowconfigure(8, weight=1)
        
        # Logo/Title
        logo_label = ctk.CTkLabel(
            sidebar,
            text="⚡ NeuraForge",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        subtitle = ctk.CTkLabel(
            sidebar,
            text="Unified AI Interface",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        subtitle.grid(row=1, column=0, padx=20, pady=(0, 20))
        
        # Mode buttons
        self.text_btn = ctk.CTkButton(
            sidebar,
            text="📝 Text Generation",
            command=lambda: self.switch_mode("text"),
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.text_btn.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        self.image_btn = ctk.CTkButton(
            sidebar,
            text="🎨 Image Generation",
            command=lambda: self.switch_mode("image"),
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.image_btn.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        # Model Management Section
        model_label = ctk.CTkLabel(
            sidebar,
            text="Model Management",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        model_label.grid(row=4, column=0, padx=20, pady=(30, 10))
        
        # Model selector
        self.model_var = ctk.StringVar(value="No model loaded")
        self.model_dropdown = ctk.CTkOptionMenu(
            sidebar,
            variable=self.model_var,
            values=["No models found"],
            command=self.on_model_selected
        )
        self.model_dropdown.grid(row=5, column=0, padx=20, pady=10, sticky="ew")
        
        # Load model button
        self.load_btn = ctk.CTkButton(
            sidebar,
            text="🔄 Load Selected Model",
            command=self.load_selected_model,
            height=35
        )
        self.load_btn.grid(row=6, column=0, padx=20, pady=10, sticky="ew")
        
        # Refresh models button
        refresh_btn = ctk.CTkButton(
            sidebar,
            text="♻️ Refresh Models",
            command=self.refresh_models,
            height=35,
            fg_color="transparent",
            border_width=2
        )
        refresh_btn.grid(row=7, column=0, padx=20, pady=10, sticky="ew")
        
        # Status label
        self.status_label = ctk.CTkLabel(
            sidebar,
            text=f"GPU: {self.model_manager.device.upper()}",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.status_label.grid(row=9, column=0, padx=20, pady=(0, 20))
        
        # Initial model refresh
        self.refresh_models()
    
    def create_text_interface(self):
        """Create text generation interface"""
        
        self.text_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        self.text_frame.grid_columnconfigure(0, weight=1)
        self.text_frame.grid_rowconfigure(2, weight=1)
        
        # Title
        title = ctk.CTkLabel(
            self.text_frame,
            text="Text Generation",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        
        # Input section
        input_label = ctk.CTkLabel(
            self.text_frame,
            text="Prompt:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        input_label.grid(row=1, column=0, padx=20, pady=(10, 5), sticky="w")
        
        self.text_input = ctk.CTkTextbox(
            self.text_frame,
            height=100,
            font=ctk.CTkFont(size=13)
        )
        self.text_input.grid(row=2, column=0, padx=20, pady=(0, 10), sticky="nsew")
        
        # Parameters frame
        params_frame = ctk.CTkFrame(self.text_frame, fg_color="transparent")
        params_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        params_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Max length
        length_label = ctk.CTkLabel(params_frame, text="Max Length:")
        length_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.max_length = ctk.CTkSlider(
            params_frame,
            from_=50,
            to=1000,
            number_of_steps=19
        )
        self.max_length.set(200)
        self.max_length.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        # Temperature
        temp_label = ctk.CTkLabel(params_frame, text="Temperature:")
        temp_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.temperature = ctk.CTkSlider(
            params_frame,
            from_=0.1,
            to=2.0,
            number_of_steps=19
        )
        self.temperature.set(0.7)
        self.temperature.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        
        # Generate button
        self.text_gen_btn = ctk.CTkButton(
            self.text_frame,
            text="🚀 Generate Text",
            command=self.generate_text,
            height=45,
            font=ctk.CTkFont(size=15, weight="bold")
        )
        self.text_gen_btn.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        
        # Output section
        output_label = ctk.CTkLabel(
            self.text_frame,
            text="Generated Output:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        output_label.grid(row=5, column=0, padx=20, pady=(20, 5), sticky="w")
        
        self.text_output = ctk.CTkTextbox(
            self.text_frame,
            height=200,
            font=ctk.CTkFont(size=13)
        )
        self.text_output.grid(row=6, column=0, padx=20, pady=(0, 20), sticky="nsew")
    
    def create_image_interface(self):
        """Create image generation interface"""
        
        self.image_frame = ctk.CTkFrame(self.main_frame, corner_radius=10)
        self.image_frame.grid_columnconfigure((0, 1), weight=1)
        self.image_frame.grid_rowconfigure(6, weight=1)
        
        # Title
        title = ctk.CTkLabel(
            self.image_frame,
            text="Image Generation",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="w")
        
        # Prompt
        prompt_label = ctk.CTkLabel(
            self.image_frame,
            text="Prompt:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        prompt_label.grid(row=1, column=0, columnspan=2, padx=20, pady=(10, 5), sticky="w")
        
        self.img_prompt = ctk.CTkTextbox(
            self.image_frame,
            height=80,
            font=ctk.CTkFont(size=13)
        )
        self.img_prompt.grid(row=2, column=0, columnspan=2, padx=20, pady=(0, 10), sticky="ew")
        
        # Negative prompt
        neg_label = ctk.CTkLabel(
            self.image_frame,
            text="Negative Prompt:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        neg_label.grid(row=3, column=0, columnspan=2, padx=20, pady=(10, 5), sticky="w")
        
        self.img_neg_prompt = ctk.CTkTextbox(
            self.image_frame,
            height=60,
            font=ctk.CTkFont(size=13)
        )
        self.img_neg_prompt.grid(row=4, column=0, columnspan=2, padx=20, pady=(0, 10), sticky="ew")
        
        # Parameters
        params_frame = ctk.CTkFrame(self.image_frame, fg_color="transparent")
        params_frame.grid(row=5, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        params_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        # Steps
        steps_label = ctk.CTkLabel(params_frame, text="Steps:")
        steps_label.grid(row=0, column=0, padx=5, pady=5)
        self.steps = ctk.CTkSlider(params_frame, from_=10, to=100, number_of_steps=18)
        self.steps.set(30)
        self.steps.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Guidance
        guidance_label = ctk.CTkLabel(params_frame, text="Guidance:")
        guidance_label.grid(row=0, column=2, padx=5, pady=5)
        self.guidance = ctk.CTkSlider(params_frame, from_=1, to=20, number_of_steps=19)
        self.guidance.set(7.5)
        self.guidance.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        
        # Width
        width_label = ctk.CTkLabel(params_frame, text="Width:")
        width_label.grid(row=1, column=0, padx=5, pady=5)
        self.width = ctk.CTkOptionMenu(
            params_frame,
            values=["512", "768", "1024"]
        )
        self.width.set("512")
        self.width.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        # Height
        height_label = ctk.CTkLabel(params_frame, text="Height:")
        height_label.grid(row=1, column=2, padx=5, pady=5)
        self.height = ctk.CTkOptionMenu(
            params_frame,
            values=["512", "768", "1024"]
        )
        self.height.set("512")
        self.height.grid(row=1, column=3, padx=5, pady=5, sticky="ew")
        
        # Generate button
        self.img_gen_btn = ctk.CTkButton(
            self.image_frame,
            text="🎨 Generate Image",
            command=self.generate_image,
            height=45,
            font=ctk.CTkFont(size=15, weight="bold")
        )
        self.img_gen_btn.grid(row=6, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        
        # Image preview
        self.image_label = ctk.CTkLabel(
            self.image_frame,
            text="Generated image will appear here",
            fg_color="gray20",
            corner_radius=10
        )
        self.image_label.grid(row=7, column=0, columnspan=2, padx=20, pady=(10, 20), sticky="nsew")
    
    def switch_mode(self, mode: str):
        """Switch between text and image generation modes"""
        self.current_mode = mode
        
        # Update button styles
        if mode == "text":
            self.text_btn.configure(fg_color=("gray75", "gray25"))
            self.image_btn.configure(fg_color=["#3B8ED0", "#1F6AA5"])
            self.text_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
            self.image_frame.grid_forget()
        else:
            self.image_btn.configure(fg_color=("gray75", "gray25"))
            self.text_btn.configure(fg_color=["#3B8ED0", "#1F6AA5"])
            self.image_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
            self.text_frame.grid_forget()
        
        self.refresh_models()
    
    def refresh_models(self):
        """Refresh available models list"""
        model_type = "text-generation" if self.current_mode == "text" else "image-generation"
        models = self.model_manager.get_available_models(model_type)
        
        if models:
            model_names = [f"{m['name']} ({m['size']})" for m in models]
            self.model_dropdown.configure(values=model_names)
            self.model_var.set(model_names[0])
        else:
            self.model_dropdown.configure(values=["No models found"])
            self.model_var.set("No models found")
    
    def on_model_selected(self, selection):
        """Handle model selection"""
        pass
    
    def load_selected_model(self):
        """Load the selected model"""
        selection = self.model_var.get()
        
        if selection == "No models found":
            self.show_status("No models available", error=True)
            return
        
        # Extract model name from selection
        model_name = selection.split(" (")[0]
        model_type = "text-generation" if self.current_mode == "text" else "image-generation"
        model_path = MODELS_PATH / model_type / model_name
        
        # Disable button during loading
        self.load_btn.configure(state="disabled", text="⏳ Loading...")
        
        def load_thread():
            def update_status(msg):
                self.after(0, lambda: self.show_status(msg))
            
            if self.current_mode == "text":
                success = self.model_manager.load_text_model(str(model_path), update_status)
            else:
                success = self.model_manager.load_image_model(str(model_path), update_status)
            
            self.after(0, lambda: self.load_btn.configure(
                state="normal",
                text="🔄 Load Selected Model"
            ))
            
            if success:
                self.after(0, lambda: self.show_status(f"✓ Model loaded: {model_name}"))
            else:
                self.after(0, lambda: self.show_status("✗ Failed to load model", error=True))
        
        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()
    
    def generate_text(self):
        """Generate text using loaded model"""
        prompt = self.text_input.get("1.0", "end-1c").strip()
        
        if not prompt:
            self.show_status("Please enter a prompt", error=True)
            return
        
        self.text_gen_btn.configure(state="disabled", text="⏳ Generating...")
        self.text_output.delete("1.0", "end")
        self.text_output.insert("1.0", "Generating response...\n")
        
        def gen_thread():
            max_len = int(self.max_length.get())
            temp = self.temperature.get()
            
            response = self.model_manager.generate_text(prompt, max_len, temp)
            
            self.after(0, lambda: self.text_output.delete("1.0", "end"))
            self.after(0, lambda: self.text_output.insert("1.0", response))
            self.after(0, lambda: self.text_gen_btn.configure(
                state="normal",
                text="🚀 Generate Text"
            ))
            self.after(0, lambda: self.show_status("✓ Text generated"))
            
            # Save output
            self.save_text_output(prompt, response)
        
        thread = threading.Thread(target=gen_thread, daemon=True)
        thread.start()
    
    def generate_image(self):
        """Generate image using loaded model"""
        prompt = self.img_prompt.get("1.0", "end-1c").strip()
        
        if not prompt:
            self.show_status("Please enter a prompt", error=True)
            return
        
        self.img_gen_btn.configure(state="disabled", text="⏳ Generating...")
        
        def gen_thread():
            neg_prompt = self.img_neg_prompt.get("1.0", "end-1c").strip()
            steps = int(self.steps.get())
            guidance = self.guidance.get()
            width = int(self.width.get())
            height = int(self.height.get())
            
            image, msg = self.model_manager.generate_image(
                prompt, neg_prompt, steps, guidance, width, height
            )
            
            self.after(0, lambda: self.img_gen_btn.configure(
                state="normal",
                text="🎨 Generate Image"
            ))
            
            if image:
                # Save and display image
                output_path = self.save_image_output(image, prompt)
                self.after(0, lambda: self.display_image(image))
                self.after(0, lambda: self.show_status(f"✓ Image saved: {output_path.name}"))
            else:
                self.after(0, lambda: self.show_status(msg, error=True))
        
        thread = threading.Thread(target=gen_thread, daemon=True)
        thread.start()
    
    def display_image(self, pil_image):
        """Display generated image"""
        # Resize for display
        display_size = (400, 400)
        img_copy = pil_image.copy()
        img_copy.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        ctk_image = ctk.CTkImage(
            light_image=img_copy,
            dark_image=img_copy,
            size=img_copy.size
        )
        
        self.image_label.configure(image=ctk_image, text="")
        self.image_label.image = ctk_image
    
    def save_text_output(self, prompt: str, response: str):
        """Save text generation output"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUTS_PATH / "text" / f"text_{timestamp}.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"PROMPT:\n{prompt}\n\n")
            f.write(f"RESPONSE:\n{response}\n")
    
    def save_image_output(self, image: Image.Image, prompt: str) -> Path:
        """Save image generation output"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUTS_PATH / "images" / f"image_{timestamp}.png"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save image
        image.save(output_file)
        
        # Save metadata
        metadata_file = output_file.with_suffix('.txt')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write(f"PROMPT:\n{prompt}\n")
        
        return output_file
    
    def show_status(self, message: str, error: bool = False):
        """Show status message"""
        color = "red" if error else "gray"
        self.status_label.configure(text=message, text_color=color)
        
        # Reset after 5 seconds
        self.after(5000, lambda: self.status_label.configure(
            text=f"GPU: {self.model_manager.device.upper()}",
            text_color="gray"
        ))


def main():
    """Main entry point"""
    
    # Ensure directories exist
    for path in [MODELS_PATH / "text-generation", MODELS_PATH / "image-generation", 
                 OUTPUTS_PATH / "text", OUTPUTS_PATH / "images"]:
        path.mkdir(parents=True, exist_ok=True)
    
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║              NeuraForge - Starting Up                ║
    ║         Unified AI Model Interface v1.0              ║
    ║                                                       ║
    ╚═══════════════════════════════════════
