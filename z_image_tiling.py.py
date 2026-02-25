import os, torch, random, gc
from typing import Optional
from torch import Tensor, nn
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
from tqdm import tqdm

class ZImageTiler:
    def __init__(self, model_path: str, device: str = "cuda", torch_dtype=torch.bfloat16):
        """
        Initialize the Z-Image pipeline with seamless tiling capabilities.
        :param model_path: Base directory for DiffSynth models.
        """
        os.environ['DIFFSYNTH_MODEL_BASE_PATH'] = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Load the Z-Image Pipeline
        self.pipe = ZImagePipeline.from_pretrained(
            torch_dtype=torch_dtype, device=device,
            model_configs=[
                ModelConfig(model_id="Tongyi-MAI/Z-Image", origin_file_pattern="transformer/*.safetensors"),
                ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors"),
                ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
            ],
            tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
        )

    def _set_tiling_mode(self, enable: bool):
        """
        Inject circular padding into VAE Conv2d layers to ensure seamless boundaries.
        """
        mode = 'circular' if enable else 'constant'
        for layer in self.pipe.vae_decoder.modules():
            if isinstance(layer, nn.Conv2d):
                layer.padding_mode = mode

    @torch.no_grad()
    def generate(self, prompt: str, seed: int = 42, steps: int = 50, cfg: float = 4.0, size: int = 1024):
        """
        Generate a seamless image using noise rolling and circular VAE padding.
        """
        # 1. Setup seeds and initialize latents
        random.seed(seed)
        torch.manual_seed(seed)
        
        # Use pipe's internal runner to prepare inputs (prompt encoding, latent init)
        inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
            self.pipe.units[0], self.pipe, 
            {"cfg_scale": cfg, "height": size, "width": size, "seed": seed, "num_inference_steps": steps},
            {"prompt": prompt}, {"negative_prompt": ""}
        )
        
        # 2. Sampling loop with Noise Rolling
        self.pipe.load_models_to_device(self.pipe.in_iteration_models)
        latents = inputs_shared["latents"]
        _, _, h, w = latents.shape

        for i, t in enumerate(tqdm(self.pipe.scheduler.timesteps, desc="Sampling")):
            timestep = t.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            # Randomly roll latents to hide edge artifacts
            sx, sy = random.randint(0, w - 1), random.randint(0, h - 1)
            inputs_shared["latents"] = torch.roll(latents, (sy, sx), (2, 3))
            
            # Predict noise and un-roll the prediction
            noise_pred = self.pipe.cfg_guided_model_fn(
                self.pipe.model_fn, cfg, inputs_shared, 
                inputs_posi, inputs_nega, timestep=timestep, progress_id=i
            )
            noise_pred = torch.roll(noise_pred, (-sy, -sx), (2, 3))
            
            # Denoising step
            inputs_shared["latents"] = latents
            latents = self.pipe.step(self.pipe.scheduler, progress_id=i, noise_pred=noise_pred, **inputs_shared)

        # 3. Seamless VAE Decoding
        self.pipe.load_models_to_device(['vae_decoder'])
        self._set_tiling_mode(True)
        
        image = self.pipe.vae_output_to_image(self.pipe.vae_decoder(latents))
        
        self._set_tiling_mode(False) # Restore state
        self.pipe.load_models_to_device([]) # Offload models
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        return image

if __name__ == "__main__":
    # Example Usage
    tiler = ZImageTiler(model_path=model_path)
    img = tiler.generate(
        prompt="texture of potatoes surface",
        seed=336844923011399,
        steps=50
    )
    img.save("seamless_texture.png")