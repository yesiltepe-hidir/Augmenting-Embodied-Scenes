import torch
import abc
from typing import Any, Dict, Optional, Type
import torch.nn as nn
import numpy as np
import os 

def batch_cosine_sim(x, y):
    if type(x) is list:
        x = torch.cat(x, dim=0)
    if type(y) is list:
        y = torch.cat(y, dim=0)
    x_ = x / x.norm(dim=-1, keepdim=True) # 1 x D
    y_ = y / y.norm(dim=-1, keepdim=True) # N x D
    similarity = (x_ @ y_.T).squeeze() # N
    idx = torch.argmax(similarity)
    return y[idx], similarity

def flatten_grid(x, grid_size=[2,2]): 
    ''' 
    x: B x C x H x W
    '''
    B, C, H, W = x.size()
    hs = grid_size[0]
    ws = grid_size[1]
    img_h = H // hs
    img_w = W // ws

    flattened = torch.zeros((B, C, img_h, hs*img_w*ws), device=x.device)
    k = 0
    for h in range(hs):
        for w in range(ws):
            flattened[:,:,:,k*img_w:(k+1)*img_w] = x[:,:,h*img_h:(h+1)*img_h,w*img_w:(w+1)*img_w]
            k += 1
        
    return flattened

def unflatten_grid(x, grid_size=[2,2]):
    ''' 
    x: B x C x H x W
    '''
    B, C, H, W = x.size()
    hs = grid_size[0]
    ws = grid_size[1]
    img_w = W // (ws * hs)
    img_h = H

    unflattened = torch.zeros((B, C, img_h * hs, img_w * ws), device=x.device)
    k = 0
    for h in range(hs):
        for w in range(ws):
            unflattened[:,:,h*img_h:(h+1)*img_h,w*img_w:(w+1)*img_w] = x[:,:,:,k*img_w:(k+1)*img_w]
            k += 1
        
    return unflattened
    

def prepare_key_grid_latents(latents_video, latent_grid_size=[2,2], key_grid_size=[3,3], rand_indices=None):
    '''
    latents_video: T x C x H x W
    '''
    img_h, img_w = latents_video.size(-2) // latent_grid_size[0], latents_video.size(-1) // latent_grid_size[1]
    concat_all = torch.cat([flatten_grid(el.unsqueeze(0), latent_grid_size) for el in latents_video], dim=-1)
    keyframe_grid = unflatten_grid(torch.cat([concat_all[:,:,:,ind.item()*(img_w):(ind.item()+1)*(img_w)] for ind in rand_indices.squeeze()], dim=-1), key_grid_size)
    return keyframe_grid, rand_indices
    
def pil_grid_to_frames(pil_grid, grid_size=[2,2]):
    w,h = pil_grid.size
    img_w = w // grid_size[1]
    img_h = h // grid_size[0]
    list_of_pil = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            list_of_pil.append(pil_grid.crop((j*img_w, i*img_h, (j+1)*img_w, (i+1)*img_h)))
    return list_of_pil
    
class FeatureFlow(abc.ABC):

    def __call__(self, tokens, place_in_unet):
        h = tokens.shape[0]
        print(tokens.shape)
        tokens[h // 2:] = self.forward(tokens[h // 2:], place_in_unet)
        return tokens
    
    def forward(self, tokens, place_in_unet):
        self.tokens.append(tokens.detach().cpu())
        return tokens
    
    def __init__(self):
        self.tokens = []


def register_feature_controller(model, controller):
    def sa_forward(self, place_in_unet, prev_hidden_states=None):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,):
            is_cross = encoder_hidden_states is not None
            
            residual = hidden_states

            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = to_out(hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor
            if not is_cross:
                hidden_states = controller(hidden_states, place_in_unet)

            return hidden_states
        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_.forward = sa_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


class BaseFeatures(abc.ABC):
    def __init__(self):
        pass
    
    def __call__(self):
        pass
    
    def empty_features(self):
        pass

'''
class StoreFeatures(BaseFeatures):
    def __init__(self):
        self.features = []
    
    def __call__(self, attn_output):
        
        self.features.append(attn_output.detach().cpu())

    def empty_features(self):
        self.features = []
'''

class StoreFeatures(BaseFeatures):
    def __init__(self):
        self.self_attn_inputs    = [[], []]
        self.method_type         = ""   
        self.task                = ""
        self.sample_size         = 1
        self.sample_idx          = 0    
        self.step                = 0
        self.num_attn_layers     = 16
        self.inference_steps     = 20 
        self.curr_inference_step = 0
         
    def __call__(self, attn):  
        if self.task == "ddim_inversion":
            return

        # Extended Attention
        if self.method_type == "extended_attention":
            self.self_attn_inputs[self.sample_idx].append(attn.detach().cpu())
    
    def between(self):
        if self.task == "ddim_inversion":
            return
        # Incerement the current step size each time
        self.step += 1
        # When all attention blocks are done, incerement the sample index
        if self.step == self.num_attn_layers:
            self.step = 0
            self.curr_inference_step += 1
            # If 1 timestep is completed then start from the beginning
            if len(self.self_attn_inputs[-1]) == self.inference_steps * self.num_attn_layers:
                # self.empty()
                self.sample_idx += 1 
                if self.method_type == "extended_attention":
                    self.self_attn_inputs.append([])
                save_list = [self.sample_idx, self.self_attn_inputs]
                torch.save(save_list, '/home/grads/hidir/allenact/controlnet/store_params.pt')
                

    def empty(self):
        self.curr_inference_step += 1
        # Extended attention
        if self.method_type == "extended_attention":
            self.self_attn_inputs = [[]]

        self.sample_idx = 0
        self.step = 0
    
    def set_method_type(self, method_name):
        print("method type:", method_name)
        # Set the method name
        self.method_type = method_name
        
        # If cross frame attention, attend to all previous frames
        if self.method_type == "extended_attention":
            self.self_attn_inputs = [[]]
    
    def init(self, method_name, num_inference_steps):
        self.set_method_type(method_name)  
        self.sample_idx = 0    
        self.sample_size = 1
        self.step = 0
        self.curr_inference_step = 0
        self.inference_steps = num_inference_steps 
        print("AAAA")
        if os.path.exists('/home/grads/hidir/allenact/controlnet/store_params.pt'): 
            self.load()
            print("BBBBB")
            print("Loaded Store parameters...")
    
    def set_task(self, task):
        self.task = task
    
    def load(self):
        store_params = torch.load('/home/grads/hidir/allenact/controlnet/store_params.pt')
        self.sample_idx, self.self_attn_inputs = store_params
        self.sample_size = self.sample_idx + 1
        

class EmptyFeatures(BaseFeatures):
    def __init__(self):
        self.self_attn_inputs    = [[], []]
        self.method_type         = ""   
        self.task                = ""
        self.sample_size         = 1
        self.sample_idx          = 0    
        self.step                = 0
        self.num_attn_layers     = 16
        self.inference_steps     = 20 
        self.curr_inference_step = 0
         
    def __call__(self, attn):  
        return

    def between(self):
        return

    def set_method_type(self, method_name):
        return
    
    def set_task(self, task):
        return
    
    def init(self, method_name, num_inference_steps):
        return
    
    def empty(self):
        return

def make_my_transformer_block(block_class, controller):
    
    class FeatureFlowBlock(block_class):
        
        def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
        ):
            
            # Notice that normalization is always applied before the real computation in the following blocks.
            # 1. Self-Attention
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # Self Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            ####################################################################################################
            context = None
            
            if controller.method_type == 'extended_attention':
                if controller.task == 'denoising' and controller.sample_idx == 0:
                    controller(norm_hidden_states)
                    context =  None
                    
                elif controller.task == 'denoising' and controller.sample_idx != 0:
                    prev_attns = []
                    for sample_idx in range(controller.sample_idx):
                        prev_attns.append(controller.self_attn_inputs[sample_idx][controller.curr_inference_step * controller.num_attn_layers + controller.step].to(hidden_states.device))
                    context = torch.cat(prev_attns, dim=1)
                    # print('context:', context.shape)
                    controller(norm_hidden_states)

            ####################################################################################################
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=context,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

            controller.between()
                  
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = attn_output + hidden_states

            # 2. Cross-Attention
            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )

                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states
    return FeatureFlowBlock  

# def make_my_transformer_block(block_class, controller):
    
#     class FeatureFlowBlock(block_class):
        
#         def forward(
#             self,
#             hidden_states: torch.FloatTensor,
#             attention_mask: Optional[torch.FloatTensor] = None,
#             encoder_hidden_states: Optional[torch.FloatTensor] = None,
#             encoder_attention_mask: Optional[torch.FloatTensor] = None,
#             timestep: Optional[torch.LongTensor] = None,
#             cross_attention_kwargs: Dict[str, Any] = None,
#             class_labels: Optional[torch.LongTensor] = None,
#         ):
            
#             # Notice that normalization is always applied before the real computation in the following blocks.
#             # 1. Self-Attention
#             if self.use_ada_layer_norm:
#                 norm_hidden_states = self.norm1(hidden_states, timestep)
#             elif self.use_ada_layer_norm_zero:
#                 norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
#                     hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
#                 )
#             else:
#                 norm_hidden_states = self.norm1(hidden_states)


#             cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
#             attn_output = self.attn1(
#                 norm_hidden_states,
#                 encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
#                 attention_mask=attention_mask,
#                 **cross_attention_kwargs,
#             )
            
#             controller(attn_output)
            
            
            
            
            
#             if self.use_ada_layer_norm_zero:
#                 attn_output = gate_msa.unsqueeze(1) * attn_output
#             hidden_states = attn_output + hidden_states

#             # 2. Cross-Attention
#             if self.attn2 is not None:
#                 norm_hidden_states = (
#                     self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
#                 )

#                 attn_output = self.attn2(
#                     norm_hidden_states,
#                     encoder_hidden_states=encoder_hidden_states,
#                     attention_mask=encoder_attention_mask,
#                     **cross_attention_kwargs,
#                 )
#                 hidden_states = attn_output + hidden_states

#             # 3. Feed-forward
#             norm_hidden_states = self.norm3(hidden_states)

#             if self.use_ada_layer_norm_zero:
#                 norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

#             ff_output = self.ff(norm_hidden_states)

#             if self.use_ada_layer_norm_zero:
#                 ff_output = gate_mlp.unsqueeze(1) * ff_output

#             hidden_states = ff_output + hidden_states

#             return hidden_states
#     return FeatureFlowBlock

  
def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    return False



if __name__ == '__main__':
    a = torch.randint(0,5,(1,3), dtype=torch.float)
    b = torch.randint(0,5,(4,3), dtype=torch.float)
    y, similarity = batch_cosine_sim(a,b)
    print(y, similarity)
    print(a)
    print(b)
    
    