from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
)

model.to(device)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
png_dir = "Frames_60/main_logs/141_212_108_99_20210407_173104/play1_141_212_108_99_20210407_173104/play1_141_212_108_99_20210407_173104_0001.png"
image = Image.open(png_dir)
inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)
