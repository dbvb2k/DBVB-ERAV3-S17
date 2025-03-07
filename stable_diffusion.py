import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from tqdm.auto import tqdm
import os
from torchvision import transforms as tfms
import requests
from pathlib import Path
import safetensors.torch
from Losses import blue_loss, custom_color_loss, EPSLoss, TAPLoss, SALoss
from torchvision.models import VGG16_Weights
import numpy as np
import datetime

# Device configuration
torch_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
if "mps" == torch_device: 
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"

# Initialize models
def init_models():
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    
    # Move to device
    vae = vae.to(torch_device)
    text_encoder = text_encoder.to(torch_device)
    unet = unet.to(torch_device)
    
    return vae, tokenizer, text_encoder, unet, scheduler

# Helper functions
def latents_to_pil(vae, latents):
    try:
        # Decode the latents
        latents = (1 / 0.18215) * latents
        with torch.no_grad():
            image = vae.decode(latents).sample
        
        # Normalize to [0, 1] range
        image = (image / 2 + 0.5).clamp(0, 1)
        
        # Convert to numpy array
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        
        # Handle potential NaN values and ensure proper range
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        image = np.clip(image, 0, 1)
        
        # Convert to uint8
        images = (image * 255).round().astype("uint8")
        
        # Add debug information
        if len(images) == 0:
            print("Warning: No images generated in latents_to_pil")
            return None
        
        # Convert to PIL images
        pil_images = [Image.fromarray(img) for img in images]
        return pil_images[0] if len(pil_images) > 0 else None
        
    except Exception as e:
        print(f"Error in latents_to_pil: {str(e)}")
        print(f"Latents shape: {latents.shape if latents is not None else 'None'}")
        print(f"Device: {latents.device if latents is not None else 'None'}")
        return None

# Download style embeddings
def download_style_embeddings():
    # Get HuggingFace token
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("\nNo Hugging Face token found in environment variables.")
        print("Please get your token from https://huggingface.co/settings/tokens")
        hf_token = input("Enter your Hugging Face token: ").strip()
        if not hf_token:
            print("No token provided. Cannot download embeddings.")
            return {}

    # Reorder the dictionary to put cat-toy before samurai-jack
    style_urls = {
        "midjourney": "https://huggingface.co/sd-concepts-library/midjourney-style/resolve/main/learned_embeds.bin",
        "rahkshi-bionicle": "https://huggingface.co/sd-concepts-library/rahkshi-bionicle/resolve/main/learned_embeds.bin",
        "black-and-white-design": "https://huggingface.co/sd-concepts-library/black-and-white-design/resolve/main/learned_embeds.bin",
        "cat-toy": "https://huggingface.co/sd-concepts-library/cat-toy/resolve/main/learned_embeds.bin",
        "madhubani-art": "https://huggingface.co/sd-concepts-library/madhubani-art/resolve/main/learned_embeds.bin"
    }
    
    style_embeds = {}
    print("\nDownloading style embeddings:")
    print("="*50)
    
    # Create style_embeds directory if it doesn't exist
    os.makedirs("style_embeds", exist_ok=True)
    
    for style_name, url in style_urls.items():
        filepath = f"style_embeds/{style_name}_embeds.bin"
        
        try:
            if not os.path.exists(filepath):
                print(f"\nDownloading {style_name} style embedding...")
                
                # Enhanced headers with user token
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'application/octet-stream',
                    'Authorization': f'Bearer {hf_token}'
                }
                
                # Try to download with a session
                with requests.Session() as session:
                    response = session.get(url, headers=headers, stream=True, timeout=30)
                    
                    if response.status_code == 200:
                        total_size = int(response.headers.get('content-length', 0))
                        block_size = 1024  # 1 KB
                        
                        with open(filepath, 'wb') as f, tqdm(
                            desc=f"{style_name}",
                            total=total_size,
                            unit='iB',
                            unit_scale=True,
                            unit_divisor=1024,
                        ) as pbar:
                            for data in response.iter_content(block_size):
                                size = f.write(data)
                                pbar.update(size)
                        print(f"✓ Successfully downloaded {style_name} embedding")
                    else:
                        print(f"✗ Failed to download {style_name} embedding")
                        print(f"Status code: {response.status_code}")
                        print(f"Response text: {response.text[:200]}")  # Print first 200 chars of response
                        continue
            
            # Load the embedding
            try:
                embedding = safetensors.torch.load_file(filepath) if filepath.endswith('.safetensors') else torch.load(filepath, map_location=torch_device)
                
                # Check embedding dimensions
                if isinstance(embedding, dict):
                    tensor = embedding.get('string_to_token', embedding.get('string_to_param', next(iter(embedding.values()))))
                else:
                    tensor = embedding
                
                # Verify tensor dimensions
                if tensor.shape[-1] != 768:  # CLIP embedding dimension
                    print(f"✗ Warning: {style_name} embedding has incompatible dimension {tensor.shape[-1]}, expected 768")
                    continue
                
                style_embeds[style_name] = embedding
                print(f"✓ Successfully loaded {style_name} embedding")
                
            except Exception as e:
                print(f"✗ Error loading {style_name} embedding: {str(e)}")
                continue
                
        except Exception as e:
            print(f"✗ Error processing {style_name}: {str(e)}")
            continue
    
    print("\nStyle embedding summary:")
    print(f"- Total styles attempted: {len(style_urls)}")
    print(f"- Successfully loaded: {len(style_embeds)}")
    print("="*50 + "\n")
    
    return style_embeds

def generate_image(
    prompt,
    vae,
    tokenizer,
    text_encoder,
    unet,
    scheduler,
    style_embed=None,
    seed=42,
    guidance_scale=7.5,
    num_inference_steps=30,
    height=512,
    width=512,
    loss_type='Blue',
    loss_scale=100
):
    try:
        NUM_DASHES = 200
        print("="*NUM_DASHES)
        print(f"Generating image with {loss_type} loss:")
        print("="*NUM_DASHES)
        print(f"- Prompt: {prompt}")
        print(f"- Seed: {seed}")
        print(f"- Loss scale: {loss_scale}")
        print(f"- Image size: {height}x{width}")
        
        # Clear CUDA cache at the start
        if torch.cuda.is_available():
            print("- Using CUDA device")
            torch.cuda.empty_cache()
        
        # Move loss functions to CPU initially
        print("Initializing loss functions...")
        eps_loss = EPSLoss().cpu()
        tap_loss = TAPLoss(weights=VGG16_Weights.IMAGENET1K_V1).cpu()
        sa_loss = SALoss().cpu()
        
        print("Encoding prompt...")
        generator = torch.manual_seed(seed)
        text_input = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        
        if style_embed is not None:
            print("Applying style embedding...")
            print(f"Style embedding type: {type(style_embed)}")
            if isinstance(style_embed, dict):
                # Debug style embedding dictionary
                print(f"Style embedding keys: {list(style_embed.keys())}")
                style_tensor = style_embed.get('string_to_token', style_embed.get('string_to_param', next(iter(style_embed.values()))))
                if style_tensor is None:
                    raise ValueError("Could not extract style tensor from embedding dictionary")
                style_tensor = style_tensor.to(torch_device)
            else:
                style_tensor = style_embed.to(torch_device)
            text_embeddings = text_embeddings + style_tensor
        
        print("Setting up unconditional guidance...")
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer([""] * 1, padding="max_length", max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        print("Initializing denoising process...")
        scheduler.set_timesteps(num_inference_steps)
        
        latents = torch.randn(
            (1, unet.config.in_channels, height // 8, width // 8),
            generator=generator,
        ).to(torch_device)
        latents = latents * scheduler.init_noise_sigma
        
        # Track loss values for analysis
        loss_history = []
        current_loss = 0.0  # Initialize current loss
        
        # Denoising loop
        print("\nStarting denoising loop:")
        progress_bar = tqdm(enumerate(scheduler.timesteps), desc=f"Denoising (Loss: N/A)", total=len(scheduler.timesteps))
        for i, t in progress_bar:
            # Clear CUDA cache periodically
            if i % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Apply loss guidance every 5 steps
            if i % 5 == 0:
                try:
                    latents = latents.detach().requires_grad_()
                    latents_x0 = latents - scheduler.sigmas[i] * noise_pred
                    
                    # Use the new autocast syntax
                    with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        denoised_images = vae.decode((1 / 0.18215) * latents_x0).sample / 2 + 0.5
                        
                        # Select loss function and compute on appropriate device
                        if loss_type == 'Blue':
                            loss = blue_loss(denoised_images) * loss_scale
                            current_loss = loss.item()
                        elif loss_type == 'Color':
                            loss = custom_color_loss(denoised_images) * loss_scale
                            current_loss = loss.item()
                        elif loss_type == 'EPS':
                            eps_loss = eps_loss.to(torch_device)
                            loss = eps_loss(denoised_images) * loss_scale
                            current_loss = loss.item()
                            eps_loss = eps_loss.cpu()
                        elif loss_type == 'TAP':
                            tap_loss = tap_loss.to(torch_device)
                            loss = tap_loss(denoised_images) * loss_scale
                            current_loss = loss.item()
                            tap_loss = tap_loss.cpu()
                        elif loss_type == 'SA':
                            sa_loss = sa_loss.to(torch_device)
                            loss = sa_loss(denoised_images) * loss_scale
                            current_loss = loss.item()
                            sa_loss = sa_loss.cpu()
                        
                        loss_history.append(current_loss)
                        # Update progress bar description with current loss
                        progress_bar.set_description(f"Denoising (Step: {i}/{len(scheduler.timesteps)}, Loss: {current_loss:.4f})")
                    
                    cond_grad = torch.autograd.grad(loss, latents)[0]
                    latents = latents.detach() - cond_grad * scheduler.sigmas[i]**2
                    
                except RuntimeError as e:
                    print(f"\nWarning: Loss computation failed at step {i}: {str(e)}")
                    continue
            else:
                # Update progress bar with step info even when not computing loss
                progress_bar.set_description(f"Denoising (Step: {i}/{len(scheduler.timesteps)}, Loss: {current_loss:.4f})")
            
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            # Clear some memory
            del noise_pred, noise_pred_uncond, noise_pred_text
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Final cleanup and statistics
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\nGeneration complete!")
        if loss_history:
            print(f"Final loss: {loss_history[-1]:.4f}")
            print(f"Average loss: {sum(loss_history)/len(loss_history):.4f}")
            print(f"Min loss: {min(loss_history):.4f}")
            print(f"Max loss: {max(loss_history):.4f}")
        
        result = latents_to_pil(vae, latents)
        if result is None:
            raise ValueError("Failed to generate image: latents_to_pil returned None")
        return result
        
    except Exception as e:
        print(f"\nError in generate_image: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

def main():
    print("="*200)
    print("Starting the training process... ")
    print("="*200)
    print("Initializing models...")
    vae, tokenizer, text_encoder, unet, scheduler = init_models()
    
    print("Downloading style embeddings...")
    style_embeds = download_style_embeddings()
    
    if not style_embeds:
        print("No style embeddings were successfully loaded. Exiting...")
        return
    
    # Define prompts
    prompts = [
        "A serene landscape with mountains and a lake",
        "A toy cat with contrasting colors",
        "Iron man and catwoman in traditional attire"
    ]
    
    # Create base output directory
    base_output_dir = "generated_images"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Loss types and their scales
    loss_types = ['Blue', 'Color', 'EPS', 'TAP', 'SA']
    loss_scales = {
        'Blue': 100.0,
        'Color': 100.0,
        'EPS': 0.1,
        'TAP': 0.5,
        'SA': 1.0
    }
    
    # Generate images for each prompt
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {prompt_idx + 1}: {prompt}")
        
        # Create prompt-specific directory (using first few words of prompt)
        # Remove 'with' and other common words, and limit to first 4 significant words
        words = prompt.lower().split()
        significant_words = [w for w in words if w not in ['with', 'and', 'the', 'a', 'an', 'in', 'on', 'at']][:4]
        prompt_dir = "_".join(significant_words)
        prompt_path = os.path.join(base_output_dir, f"prompt_{prompt_idx + 1}_{prompt_dir}", timestamp)
        
        # Generate images with different styles and loss functions
        print("Generating styled images...")
        seeds = [42, 123, 456, 789, 101112]
        
        for i, (style_name, style_embed) in enumerate(style_embeds.items()):
            print(f"\nProcessing {style_name} style...")
            try:
                # Create style-specific directories
                style_base_path = os.path.join(prompt_path, style_name)
                
                # Create directories for each loss type
                for loss_type in loss_types:
                    loss_dir = os.path.join(style_base_path, loss_type.lower())
                    os.makedirs(loss_dir, exist_ok=True)
                
                # Generate images with each loss type
                for loss_type in loss_types:
                    # print(f"Generating with {loss_type} loss...")
                    img = generate_image(
                        prompt,
                        vae,
                        tokenizer,
                        text_encoder,
                        unet,
                        scheduler,
                        style_embed=style_embed,
                        seed=seeds[i],
                        loss_type=loss_type,
                        loss_scale=loss_scales[loss_type]
                    )
                    
                    # Save to appropriate directory
                    output_path = os.path.join(style_base_path, loss_type.lower(), f"{style_name}.png")
                    img.save(output_path)
                    print(f"Successfully generated {style_name} image with {loss_type} loss")
                    
                # Create a summary file for this style
                with open(os.path.join(style_base_path, "generation_info.txt"), "w") as f:
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Style: {style_name}\n")
                    f.write(f"Seed: {seeds[i]}\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write("\nLoss Scales:\n")
                    for loss_type, scale in loss_scales.items():
                        f.write(f"{loss_type}: {scale}\n")
                
            except Exception as e:
                print(f"Error generating {style_name} images: {str(e)}")
                continue
        
        # Create a summary file for this prompt
        with open(os.path.join(prompt_path, "prompt_info.txt"), "w") as f:
            f.write(f"Generation Summary\n")
            f.write("="*50 + "\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total styles processed: {len(style_embeds)}\n")
            f.write(f"Loss types used: {', '.join(loss_types)}\n")
            f.write("\nDirectory Structure:\n")
            f.write(f"Base path: {prompt_path}\n")
            f.write("Subdirectories:\n")
            for style in style_embeds.keys():
                f.write(f"- {style}/\n")
                for loss in loss_types:
                    f.write(f"  - {loss.lower()}/\n")

if __name__ == "__main__":
    main() 