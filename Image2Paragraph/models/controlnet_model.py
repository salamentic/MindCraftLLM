import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)


class TextToImage:
    def __init__(self, device):
        self.device = device
        self.model = self.initialize_model()

    def initialize_model(self):
        if self.device == 'cpu':
            self.data_type = torch.float32
        else:
            self.data_type = torch.float16
        controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-canny",
            torch_dtype=self.data_type,
            map_location=self.device,  # Add this line
        ).to(self.device)
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            # "pretrained_models/stable-diffusion-v1-5",
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=self.data_type,
            map_location=self.device,  # Add this line
        )
        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        pipeline.to(self.device)
        if self.device != 'cpu':
            pipeline.enable_model_cpu_offload()
        return pipeline


    @staticmethod
    def preprocess_image(image):
        image = np.array(image)
        low_threshold = 100
        high_threshold = 200
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = np.stack([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image

    def text_to_image(self, text, image):
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        print('\nStep5, Text to Image:')
        image = self.preprocess_image(image)
        generated_image = self.model(text, image, num_inference_steps=20).images[0]
        print("Generated image has been svaed.")
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        return generated_image
    
    def text_to_image_debug(self, text, image):
        print("text_to_image_debug")
        return image