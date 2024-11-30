
import torch, cv2, base64
import numpy as np
from PIL import Image

from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
import litserve as ls
from io import BytesIO

from datetime import datetime

class ControlNetLitAPI(ls.LitAPI):
    def setup(self, device):
        self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
        self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        )
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            # "stabilityai/stable-diffusion-xl-base-1.0",
            "Lykon/dreamshaper-xl-v2-turbo",
            controlnet=controlnet,
            vae=vae,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        )
        self.pipe.enable_model_cpu_offload()

    def get_depth_map(self, image):
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = self.depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(720, 1280),
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

    def decode_request(self, request):
        print("Request received, decoding...")
        prompt = request["prompt"]
        image_data = base64.b64decode(request["image_base64"])
        image = Image.open(BytesIO(image_data))

        width = 720#request["width"]
        height = 1280#request["height"]
        return prompt, image, width, height

    def predict(self, inputs):
        prompt, image, width, height = inputs

        depth_image = self.get_depth_map(image)
        # depth_image.save("depth.png")

        # prompt = "hairy humanoid werewolf, bloody and scarred, imax quality, wearing torn clothes, dark and raining outside"
        prompt = "humnaoid, scars, mutant, horrific, monster, terrifying humanoid, imax, film grain, shabby hair, terrifying, raining , night time, dark brown fur, evil black eyes bloody flesh scars, img_2354, mov_7892, ((werewolf face)), werewolf, imax, film grain, wolf, shabby hair, terrifying, furry werewolf , hairy body, raining , night time, dark brown fur, evil black eyes bloody flesh scars, img_2354, mov_7892"
        negative_prompt = "((black and white)),, ((monochrome)), ((border)) out of frame, two heads, totem pole, several faces, extra fingers, mutated hands, extra hands, extra feet, (poorly drawn hands:1.21), (poorly drawn face:1.21), (mutation:1.331), (deformed:1.331), (ugly: 1.21), blurry, (bad anatomy:1.21), (bad proportions:1.331), (extra limbs:1.21), cloned face, (anime:1.331), (skinny: 1.331), glitchy, saturated colors, distorted fingers, oversaturation, (penis:1.3), 3d, cartoon, low-res, text, error, cropped, worst quality, low quality, jpeg artifacts, duplicate, morbid, mutilated, out of frame ,extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, image compression, compression, noise, closeup, flat, cartoon, 3d, (disfigured), (bad art), (deformed), (poorly drawn), (extra limbs), (close up),strange colors, blurry, boring, sketch, lackluster, cartoon, 3d, (disfigured), (bad art), (deformed), (poorly drawn), (extra limbs), (close up), strange colors, blurry, boring, sketch, lackluster, High pass filter,, (lout of focus body)), ((out of focus face)), (((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, (poorly drawn hands)), (poorly drawn face)), ((mutation))), ((deformed) |), ((ugly)), blurry, ((bad anatomy)), (((bad proportions) |), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), ((missing feet))), ((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), ((long neck))), ugly, man, (headshot:1.3), child, (closeup:1.3), fat, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, (deformed body:1.3)"
        controlnet_conditioning_scale = 0.3

        output = self.pipe(
            prompt, 
            negative_prompt=negative_prompt,
            image=depth_image, 
            num_inference_steps=30, 
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            width=width,
            height=height,
            strength=0.5
        ).images[0]

        return output
        # except RuntimeError as e:
        #     if "CUDA out of memory" in str(e):
                 

    def encode_response(self, output):
        buffered = BytesIO()
        output.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"generated_image_base64": img_str}

if __name__ == "__main__":
    api = ControlNetLitAPI()
    server = ls.LitServer(api, accelerator="auto")
    server.run(port=8000, num_api_servers=2)