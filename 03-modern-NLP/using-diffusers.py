
import torch
from diffusers import DiffusionPipeline

model_id = "segmind/tiny-sd"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading model on {device} with {dtype}...")

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=dtype,
).to(device)

prompt = "A cute robot reading a book in a library"

with torch.inference_mode():
    image = pipe(
        prompt,
        num_inference_steps=15,
    ).images[0]

image.save("generated.png")
print("Image saved as generated.png")
