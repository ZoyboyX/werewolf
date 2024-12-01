import torch  
import base64  
from io import BytesIO  
from PIL import Image  
from typing import Any  

import numpy as np
# Assuming the imports and model initialization you provided are done here.  
# For simplicity, they are reposted. Typically you would want these in a separate script and import them.  
from transformers import DPTFeatureExtractor, DPTForDepthEstimation  
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL  
from transformers import pipeline as midas_pipeline  
from diffusers.utils import load_image

# Transform code into classes/functions for handling operations  
cfg_scale = 3.5
width = 720#request["width"]
height = 1024#request["height"]
image = load_image('IMG_0881.JPG')
prompt = "humnaoid, scars, mutant, horrific, monster, terrifying humanoid, imax, film grain, shabby hair, terrifying, raining , night time, dark brown fur, evil black eyes bloody flesh scars, img_2354, mov_7892, ((werewolf face)), werewolf, imax, film grain, wolf, shabby hair, terrifying, furry werewolf , hairy body, raining , night time, dark brown fur, evil black eyes"

depth_estimator = midas_pipeline("depth-estimation")  
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")  
controlnet = ControlNetModel.from_pretrained(  
    "diffusers/controlnet-depth-sdxl-1.0",  
    variant="fp16",  
    use_safetensors=True,  
    torch_dtype=torch.float16,  
)  
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)  
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(  
    'lykon/dreamshaper-xl-turbo',  
    controlnet=controlnet,  
    vae=vae,  
    variant="fp16",  
    use_safetensors=True,  
    torch_dtype=torch.float16,  
)  

# To optimize memory usage  
pipe.enable_model_cpu_offload()  

# Use the depth estimator pipeline to get the depth map  
result = depth_estimator(image)  

# The result provides a depth map as a numpy array  
depth_map = result["depth"]  

# Ensure the depth map has 3 channels for consistency  
depth_map = np.stack([depth_map]*3, axis=-1)  

# Normalize the depth map to [0, 255]  
depth_min, depth_max = depth_map.min(), depth_map.max()  
depth_map = (depth_map - depth_min) / (depth_max - depth_min) * 255.0  

# Convert the depth map to an image object  
depth_image = Image.fromarray(depth_map.astype(np.uint8))  
depth_image.save('midasdepth.png') 

controlnet_conditioning_scale = 0.95  # recommended for good generalization  
negative_prompt = "((black and white)),, ((monochrome)), ((border)) out of frame, two heads, totem pole, several faces, extra fingers, mutated hands, extra hands, extra feet, (poorly drawn hands:1.21), (poorly drawn face:1.21), (mutation:1.331), (deformed:1.331), (ugly: 1.21), blurry, (bad anatomy:1.21), (bad proportions:1.331), (extra limbs:1.21), cloned face, (anime:1.331), (skinny: 1.331), glitchy, saturated colors, distorted fingers, oversaturation, (penis:1.3), 3d, cartoon, low-res, text, error, cropped, worst quality, low quality, jpeg artifacts, duplicate, morbid, mutilated, out of frame ,extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, image compression, compression, noise, closeup, flat, cartoon, 3d, (disfigured), (bad art), (deformed), (poorly drawn), (extra limbs), (close up),strange colors, blurry, boring, sketch, lackluster, cartoon, 3d, (disfigured), (bad art), (deformed), (poorly drawn), (extra limbs), (close up), strange colors, blurry, boring, sketch, lackluster, High pass filter,, (lout of focus body)), ((out of focus face)), (((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, (poorly drawn hands)), (poorly drawn face)), ((mutation))), ((deformed) |), ((ugly)), blurry, ((bad anatomy)), (((bad proportions) |), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), ((missing feet))), ((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), ((long neck))), ugly, man, (headshot:1.3), child, (closeup:1.3), fat, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, (deformed body:1.3)"
generated_images = pipe(  
    prompt,   
    negative_prompt=negative_prompt,
    image=depth_image,  
    guidance_scale= cfg_scale,
    width= width,
    height= height,
    num_inference_steps=30,  
    controlnet_conditioning_scale=controlnet_conditioning_scale,  
).images[0]  
generated_images.save('midasout.png')