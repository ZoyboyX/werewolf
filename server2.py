
import torch, cv2, base64
import numpy as np
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoPipelineForText2Image, DPMSolverMultistepScheduler
import litserve as ls

from datetime import datetime

class ControlNetLitAPI(ls.LitAPI):
    def setup(self, device):
        # Load ControlNet and Stable Diffusion models
        controlnet = ControlNetModel.from_pretrained(
            "TencentARC/t2i-adapter-depth-midas-sdxl-1.0",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
            device_map=None
        )
        # self.pipe = AutoPipelineForText2Image.from_pretrained(
        pipe_text2img = AutoPipelineForText2Image.from_pretrained(
            "Lykon/dreamshaper-xl-v2-turbo",
            controlnet=controlnet, 
            torch_dtype=torch.float16,
            variant="fp16"
        )
        self.pipe = AutoPipelineForText2Image.from_pipe(pipe_text2img, controlnet=controlnet)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(device)

    def decode_request(self, request):
        print("Request received, decoding...")
        prompt = request["prompt"]
        image_data = base64.b64decode(request["image"])
        image = Image.open(BytesIO(image_data)).convert("RGB")
        width = 720#request["width"]
        height = 1230#request["height"]
        return prompt, image, width, height

    def predict(self, inputs):
        prompt, image, width, height = inputs

        # DEBUG save image to verify
        # input_title = datetime.now().microsecond
        # image.save(f"{input_title}.jpg")
        
        # Preprocess the image (apply Canny edge detection)
        print(np.array(image).shape)
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)


        # Generate image
        
        # try:
        # generator = torch.Generator(self.pipe.device).manual_seed(1924556502)
        output = self.pipe(
            prompt=prompt,
            image=image,
            negative_prompt="((black and white)),, ((monochrome)), ((border)) out of frame, two heads, totem pole, several faces, extra fingers, mutated hands, extra hands, extra feet, (poorly drawn hands:1.21), (poorly drawn face:1.21), (mutation:1.331), (deformed:1.331), (ugly: 1.21), blurry, (bad anatomy:1.21), (bad proportions:1.331), (extra limbs:1.21), cloned face, (anime:1.331), (skinny: 1.331), glitchy, saturated colors, distorted fingers, oversaturation, (penis:1.3), 3d, cartoon, low-res, text, error, cropped, worst quality, low quality, jpeg artifacts, duplicate, morbid, mutilated, out of frame ,extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, image compression, compression, noise, closeup, flat, cartoon, 3d, (disfigured), (bad art), (deformed), (poorly drawn), (extra limbs), (close up),strange colors, blurry, boring, sketch, lackluster, cartoon, 3d, (disfigured), (bad art), (deformed), (poorly drawn), (extra limbs), (close up), strange colors, blurry, boring, sketch, lackluster, High pass filter,, (lout of focus body)), ((out of focus face)), (((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, (poorly drawn hands)), (poorly drawn face)), ((mutation))), ((deformed) |), ((ugly)), blurry, ((bad anatomy)), (((bad proportions) |), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), ((missing feet))), ((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), ((long neck))), ugly, man, (headshot:1.3), child, (closeup:1.3), fat, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, (deformed body:1.3)",
            num_inference_steps=35, 
            guidance_scale=4.5,
            # seed=1924556502,
            seed=2784027436,
            disable_safety_checker=True,
            width=width,
            height=height,
            controlnet_conditioning_scale=0.8,
            pixel_perfect=True
            ).images[0]  

        return output
        # except RuntimeError as e:
        #     if "CUDA out of memory" in str(e):
                 

    def encode_response(self, output):
        buffered = BytesIO()
        output.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"image": img_str}

if __name__ == "__main__":
    api = ControlNetLitAPI()
    server = ls.LitServer(api, accelerator="auto")
    server.run(port=7999, num_api_servers=2)