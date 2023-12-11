import sys
sys.path.append('/home/grads/hidir/allenact')

import torch.nn as nn
from tqdm import tqdm
from typing import Tuple
from transformers import logging
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, DDIMInverseScheduler

import controlnet.constants as const
from controlnet.attention_store import isinstance_str, make_my_transformer_block
from controlnet import utils
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

import PIL
from PIL import Image
import torch
import numpy as np
import random
from transformers import set_seed

from controlnet.attention_store import StoreFeatures


def set_seed_lib(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    set_seed(seed)

@torch.no_grad()
class StableDiffusionControlNetImg2Img(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device
        self.dtype = torch.float16

    @torch.no_grad()
    def init_pipe(self, hf_cn_path, hf_path):
        controlnet = ControlNetModel.from_pretrained(hf_cn_path, torch_dtype=self.dtype).to(self.device, self.dtype)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(hf_path, controlnet=controlnet, torch_dtype=self.dtype).to(self.device, self.dtype) 
        pipe.enable_model_cpu_offload()
        # pipe.enable_xformers_memory_efficient_attention()
        return pipe
        
    @torch.no_grad()
    def init_models(self, hf_cn_path, hf_path):
        pipe = self.init_pipe(hf_cn_path, hf_path)  
        self.vae = pipe.vae.eval()
        self._prepare_control_image = pipe.prepare_control_image
        self.run_safety_checker = pipe.run_safety_checker

        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.eval()

        self.unet = pipe.unet.eval()
        self.controlnet = pipe.controlnet.eval()
        self.scheduler_config = pipe.scheduler.config
        self.reverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
        
        del pipe
    
    @torch.no_grad()
    def set_flow_controller(self, controller):
        self.controller = controller
        for _, module in self.unet.named_modules():
            if isinstance_str(module, "BasicTransformerBlock"):
                make_tokenflow_block_fn = make_my_transformer_block 
                module.__class__ = make_tokenflow_block_fn(module.__class__, controller)
                

    
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    
    @torch.no_grad()
    def denoising_step(self, latents, control_image, text_embeddings, t, guidance_scale, current_sampling_percent):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        control_image = torch.cat([control_image] * 2)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        # compute the percentage of total steps we are at
        noise_pred = self.pred_controlnet_sampling(current_sampling_percent, latent_model_input, t, text_embeddings, control_image)

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        return latents
    

    @torch.no_grad()
    def reverse_diffusion(self, timesteps=None, latents=None, control_image=None, text_embeddings=None, height=512, width=512, num_inference_steps=50, guidance_scale=7.5):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        # self.scheduler.set_timesteps(num_inference_steps)
        with torch.autocast('cuda'):
            for i, t in tqdm(enumerate(timesteps), desc='reverse_diffusion'):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # compute the percentage of total steps we are at
                current_sampling_percent = i / len(timesteps)
                
                if (
                    current_sampling_percent < self.controlnet_guidance_start
                    or current_sampling_percent > self.controlnet_guidance_end
                ):
                    down_block_res_samples = None
                    mid_block_res_sample = None
                else:
                    
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        latent_model_input.detach(),
                        t,
                        conditioning_scale=self.controlnet_conditioning_scale,
                        encoder_hidden_states=text_embeddings,
                        controlnet_cond=control_image,
                        return_dict=False,
                    )
                

                noise_pred = self.unet(latent_model_input.detach(), t, encoder_hidden_states=text_embeddings,                    
                                    down_block_additional_residuals=down_block_res_samples,
                                    mid_block_additional_residual=mid_block_res_sample)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    @torch.no_grad()
    def encode_imgs(self, img_torch):
        '''
        Takes an image and encodes it into latents.
        ### Args:
            - `img_torch`: A torch image tensor, [batch_size, 3, height, width] in range [0, 1]
        ### Returns:
            - `latents`: [batch_size, in_channels, height//8, width//8]
            - `posterior`: Diagonal Gaussian posterior distribution
        '''
        image = 2 * img_torch - 1
        posterior = self.vae.encode(image).latent_dist
        latents = posterior.mean * self.vae.config.scaling_factor

        return latents

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor):
        '''
        Takes the latents and decodes them into an image.
        ### Args:
            - `latents`:  [batch_size, in_channels, height//8, width//8]
        ### Returns:
            - `image`: torch.Tensor, [batch_size, 3, height, width] in range [0, 1]
        '''
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    
    def set_scheduler(self, scheduler_type):
        self.scheduler = const.SCHEDULER_DICT[scheduler_type].from_config(self.scheduler_config)

    @torch.no_grad()
    def pred_controlnet_sampling(self, current_sampling_percent, latent_model_input, t, text_embeddings, control_image):
        if (
            current_sampling_percent < self.controlnet_guidance_start
            or current_sampling_percent > self.controlnet_guidance_end
        ):
            down_block_res_samples = None
            mid_block_res_sample = None
        else:
            
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input.detach(),
                t,
                conditioning_scale=self.controlnet_conditioning_scale,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=control_image,
                return_dict=False,
            )
        noise_pred = self.unet(latent_model_input.detach(), t, encoder_hidden_states=text_embeddings,                    
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample)['sample']
        return noise_pred


    @torch.no_grad()
    def ddim_inversion(self, init_latents, control_image, timesteps, num_steps, seed=0):
        
        cond = self.get_text_embeds("", "")[1].unsqueeze(0)
        latent_frames = init_latents       
        timesteps = reversed(timesteps)

        for i, t in enumerate(tqdm(timesteps)):
            curr_sampling_percent = i / len(timesteps)

            cond_batch = cond.repeat(latent_frames.shape[0], 1, 1)
                                                                
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (self.scheduler.alphas_cumprod[timesteps[i - 1]] if i > 0 else self.scheduler.final_alpha_cumprod)

            mu = alpha_prod_t ** 0.5
            mu_prev = alpha_prod_t_prev ** 0.5
            sigma = (1 - alpha_prod_t) ** 0.5
            sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

            eps = self.pred_controlnet_sampling(curr_sampling_percent, latent_frames, t, cond_batch, control_image)

            pred_x0 = (latent_frames - sigma_prev * eps) / mu_prev
            latent_frames = mu * pred_x0 + sigma * eps

        return latent_frames
    

    @torch.no_grad()
    def process_image(self, image, height, width):
        image = pil_to_tensor(image.resize((width, height))).to(self.device, self.dtype).unsqueeze(0) / 255.
        return image

    @torch.no_grad()
    def process_control_image(self, image, width, height):
        # control_image = processor(to_pil_image(image[0]))
        # control_image = control_image.resize((width, height))
        # control_image = pil_to_tensor(control_image).unsqueeze(0).to(self.device, self.dtype) / 255.
        image_processed = Image.fromarray(utils.pixel_perfect_process(np.array(image, dtype='uint8'), self.preprocess_name))
        control_image = self._prepare_control_image(
                                                image=image_processed,
                                                width=width,
                                                height=height,
                                                device=self.device,
                                                dtype=self.controlnet.dtype,
                                                batch_size=1,
                                                num_images_per_prompt=1
    ).to(self.device, self.dtype)
        
        return control_image.to(self.device)
        
    @torch.no_grad()
    def __call__(self, image, 
                       width, height, 
                       batch_size, 
                       batch_size_vae, 
                       positive_prompts,
                       num_inference_steps, 
                       guidance_scale, 
                       strength, 
                       processor, 
                       seed=0, 
                       negative_prompts='', 
                       scheduler_type=None, 
                       controlnet_conditioning_scale=1.0, 
                       controlnet_guidance_start=0.0, 
                       controlnet_guidance_end=1.0, 
                       latents_inverted=None):
                       
        set_seed_lib(seed)

        self.batch_size = batch_size
        self.batch_size_vae = batch_size_vae
        self.set_scheduler(scheduler_type)
        self.controlnet_guidance_start = controlnet_guidance_start
        self.controlnet_guidance_end = controlnet_guidance_end
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.positive_prompts = positive_prompts
        self.negative_prompts = negative_prompts
        self.preprocess_name = processor

        self.controller.init('extended_attention', num_inference_steps)

        control_image = self.process_control_image(image, width, height)
        to_pil_image(control_image[0]).save('./control_image.png')
        print('control image shape:', control_image.shape, 'max:', control_image.max())  
        image = self.process_image(image, height, width)
        text_embedding = self.get_text_embeds(positive_prompts, negative_prompts)
                    
        init_latents = self.encode_imgs(image)
        if latents_inverted is None:
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
            self.controller.set_task('ddim_inversion')
            latents_inverted = self.ddim_inversion(init_latents, control_image, timesteps, num_inference_steps, seed)
            
        else:
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
            
        self.controller.set_task('denoising')
        latents_denoised = self.reverse_diffusion(timesteps, latents_inverted, control_image, text_embedding, guidance_scale)
    
        image_torch = self.decode_latents(latents_denoised)
        image_pil = to_pil_image(image_torch[0])

        print(len(self.controller.self_attn_inputs))
        for i in self.controller.self_attn_inputs:
            print(len(i))
        
        print('#'*10)
        
        return image_pil



if __name__ == '__main__':
    from controlnet_aux import HEDdetector

    # PATHS
    IMAGE_PATH = 'debug/img_5.png'
    SAVE_PATH = './edited_image_5.png'
    
    # Get the image
    image =  Image.open(IMAGE_PATH)
    
    # Image processor
    processor = HEDdetector.from_pretrained("lllyasviel/Annotators")

    # Parameters
    params_dict = {
            'image': image,
            'width': 512,
            'height': 512,
            'batch_size': 1,
            'batch_size_vae': 1,
            'positive_prompts': 'pink indoor texture',
            'negative_prompts': '',
            'num_inference_steps': 20,
            'guidance_scale': 7.5,
            'strength': 1.0,
            'seed': 1,
            'scheduler_type': 'ddim',
            'controlnet_conditioning_scale': 1.0,
            'controlnet_guidance_start': 0.0,
            'controlnet_guidance_end': 1.0,
            'processor': processor
}

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    
    CN = StableDiffusionControlNetImg2Img(device)

    checkpoint = 'lllyasviel/control_v11p_sd15_softedge'
    hf_path = 'runwayml/stable-diffusion-v1-5'

    CN.init_models(checkpoint, hf_path)
    feature_store = StoreFeatures()
    CN.set_flow_controller(feature_store)

    edited_image = CN(**params_dict)
    edited_image.save(SAVE_PATH)