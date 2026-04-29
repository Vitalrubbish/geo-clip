import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPModel
from peft import LoraConfig, get_peft_model

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub.*')

class ImageEncoder(nn.Module):
    def __init__(self, use_lora = False):
        super(ImageEncoder, self).__init__()
        self.CLIP = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.image_processor = self._load_image_processor()
        self.mlp = nn.Sequential(nn.Linear(768, 768),
                                 nn.ReLU(),
                                 nn.Linear(768, 512))
        
        if use_lora:
            target_modules = ["v_proj", "q_proj"]
            lora_config = LoraConfig(
                r = 4,
                lora_alpha = 16,
                target_modules = target_modules,
                lora_dropout = 0.05,
                bias = "none",
            )
            self.CLIP.vision_model = get_peft_model(self.CLIP.vision_model, lora_config)
            print(f"LoRA applied to CLIP vision model with target modules: {target_modules}")
        else:
            # Freeze CLIP
            for param in self.CLIP.parameters():
                param.requires_grad = False

    def _load_image_processor(self):
        try:
            return CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14",
                local_files_only=True,
            )
        except Exception:
            return transforms.Compose(
                [
                    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )

    def preprocess_image(self, image):
        if isinstance(self.image_processor, CLIPImageProcessor):
            x = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        else:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            x = self.image_processor(image).unsqueeze(0)
        return x

    def forward(self, x):
        x = self.CLIP.get_image_features(pixel_values=x)
        # HuggingFace CLIP returns a tuple or dict, we need the tensor
        if hasattr(x, 'pooler_output'):
            x = x.pooler_output
        elif isinstance(x, tuple):
            x = x[0]
        x = self.mlp(x)
        return x