from typing import Optional
from dataclasses import dataclass
from .utils import set_seed, device_str
import torch

DEFAULT_SD = "runwayml/stable-diffusion-v1-5"

@dataclass
class ImageConfig:
    steps: int = 30
    guidance: float = 7.5
    width: int = 512
    height: int = 512
    seed: Optional[int] = None
    disable_safety: bool = False

class ImageGenerator:
    def __init__(self, model_name: str = DEFAULT_SD):
        from diffusers import StableDiffusionPipeline
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype)
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.pipe = self.pipe.to("mps")
        else:
            self.pipe = self.pipe.to("cpu")

    def generate(self, prompt: str, cfg: Optional[ImageConfig] = None):
        cfg = cfg or ImageConfig()
        seed = set_seed(cfg.seed)
        generator = torch.Generator(device=self.pipe.device).manual_seed(seed)

        safety = None
        if cfg.disable_safety:
            # Disable safety checker by replacing with lambda
            safety = self.pipe.safety_checker
            self.pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

        image = self.pipe(
            prompt=prompt,
            num_inference_steps=cfg.steps,
            guidance_scale=cfg.guidance,
            width=cfg.width,
            height=cfg.height,
            generator=generator,
        ).images[0]

        if safety is not None:
            self.pipe.safety_checker = safety
        return image, seed
