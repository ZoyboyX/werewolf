import torch  
import base64  
from io import BytesIO  
from PIL import Image  
from fastapi import FastAPI, HTTPException  
from pydantic import BaseModel  
from typing import Any  

import numpy as np
# Assuming the imports and model initialization you provided are done here.  
# For simplicity, they are reposted. Typically you would want these in a separate script and import them.  
from transformers import DPTFeatureExtractor, DPTForDepthEstimation  
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL  

# Transform code into classes/functions for handling operations  
cfg_scale = 3.5

class StableDiffusionService:  
    def __init__(self):  
        self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")  
        self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")  
        self.controlnet = ControlNetModel.from_pretrained(  
            "diffusers/controlnet-depth-sdxl-1.0",  
            variant="fp16",  
            use_safetensors=True,  
            torch_dtype=torch.float16,  
        )  
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)  
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(  
            'lykon/dreamshaper-xl-turbo',  
            controlnet=self.controlnet,  
            vae=self.vae,  
            variant="fp16",  
            use_safetensors=True,  
            torch_dtype=torch.float16,  
        )  
        
        # To optimize memory usage  
        self.pipe.enable_model_cpu_offload()  

    def get_depth_map(self, image: Image.Image) -> Image.Image:  
        image_tensor = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")  
        with torch.no_grad(), torch.autocast("cuda"):  
            depth_map = self.depth_estimator(image_tensor).predicted_depth  

        depth_map = torch.nn.functional.interpolate(  
                                                    
            depth_map.unsqueeze(1),  
            size=(1024, 720),  
            mode="bicubic",  
            align_corners=False,  
        )  
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)  
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)  
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)  
        image = torch.cat([depth_map] * 3, dim=1)  
        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]  
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))  
        return image  

    def generate_image(self, prompt: str, image: Image.Image) -> Image.Image:  
        depth_image = self.get_depth_map(image)  
        controlnet_conditioning_scale = 0.95  # recommended for good generalization  
        negative_prompt = "((black and white)),, ((monochrome)), ((border)) out of frame, two heads, totem pole, several faces, extra fingers, mutated hands, extra hands, extra feet, (poorly drawn hands:1.21), (poorly drawn face:1.21), (mutation:1.331), (deformed:1.331), (ugly: 1.21), blurry, (bad anatomy:1.21), (bad proportions:1.331), (extra limbs:1.21), cloned face, (anime:1.331), (skinny: 1.331), glitchy, saturated colors, distorted fingers, oversaturation, (penis:1.3), 3d, cartoon, low-res, text, error, cropped, worst quality, low quality, jpeg artifacts, duplicate, morbid, mutilated, out of frame ,extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, image compression, compression, noise, closeup, flat, cartoon, 3d, (disfigured), (bad art), (deformed), (poorly drawn), (extra limbs), (close up),strange colors, blurry, boring, sketch, lackluster, cartoon, 3d, (disfigured), (bad art), (deformed), (poorly drawn), (extra limbs), (close up), strange colors, blurry, boring, sketch, lackluster, High pass filter,, (lout of focus body)), ((out of focus face)), (((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, (poorly drawn hands)), (poorly drawn face)), ((mutation))), ((deformed) |), ((ugly)), blurry, ((bad anatomy)), (((bad proportions) |), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), ((missing feet))), ((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), ((long neck))), ugly, man, (headshot:1.3), child, (closeup:1.3), fat, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, (deformed body:1.3)"
        width = 720
        height = 1024
        generated_images = self.pipe(  
            prompt,   
            negative_prompt=negative_prompt,
            image=depth_image,  
            guidance_scale= cfg_scale,
            width= width,
            height= height,
            num_inference_steps=30,  
            controlnet_conditioning_scale=controlnet_conditioning_scale,  
        ).images  
        return generated_images[0]  

# Define the FastAPI app  
app = FastAPI()  

# Initialize the diffusion service  
sd_service = StableDiffusionService()  

# Create a data model for the input  
class ImageRequest(BaseModel):  
    prompt: str  
    image_base64: str  

@app.post("")  
async def generate_image_endpoint(request: ImageRequest) -> Any:  
    try:  
        # Decode the base64 image  
        image_data = base64.b64decode(request.image_base64)  
        image = Image.open(BytesIO(image_data))  

        # Generate the image using the StableDiffusionService  
        generated_image = sd_service.generate_image(request.prompt, image)  

        # Convert the generated image to a base64 string  
        buffered = BytesIO()  
        generated_image.save(buffered, format="PNG")  
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")  

        return {"generated_image_base64": img_str}  

    except Exception as e:  
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8080)