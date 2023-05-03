import torch
import os
import numpy as np
import time
from .config import Direction
from .utils import init_seed
from .lens_shift_and_zoom import self_inpaint,self_inpaint2, write_frames, zoom_in
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange, repeat
from ..ldm.util import instantiate_from_config
from ..ldm.models.diffusion.ddim import DDIMSampler
from ..ddim.diffusion import Diffusion
from ..base_model import UNetTransformerArchResBlock,UNet_Tranformer_ResBlock_Arch_Embeding
from cn_clip import clip
from cn_clip.clip import load_from_name, available_models


def inverse_data_transform(X,uniform_dequantization=False, 
                   gaussian_dequantization=False,
                  rescaled=True,
                  has_image_mean=False):
#     if has_image_mean:
#         X = X + config.image_mean.to(X.device)[None, ...]

    if False:
        X = torch.sigmoid(X)
    elif rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)


class Predictor:
    def __init__(self,
                 text_img_model_path,
                 base_dir = "easy_diffusion/predictor",
                 inpaint_yaml = 'yaml/v1-inpainting-inference.yaml',
                 inpaint_ckpt = 'ckpt/sd-v1-5-inpainting.ckpt',
                 image_shape=(3,128,128),seed=999):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        init_seed(seed)
        inpaint_conf = os.path.join(base_dir, inpaint_yaml)
        inpaint_ckpt = os.path.join(base_dir, inpaint_ckpt)
        config = OmegaConf.load(inpaint_conf)
        inpaint_model = instantiate_from_config(config.model)
        inpaint_model.load_state_dict(torch.load(inpaint_ckpt)["state_dict"], strict=False)
        inpaint_model = inpaint_model.to(self.device)
        self.inpaint_sampler = DDIMSampler(inpaint_model)
    
        model_arch = UNetTransformerArchResBlock(None,
                                                 model_channels=128,
                                                 channels=[128,128,128,256,256,512,512],
                                                 in_channels=3,
                                                 res_num=1,
                                                 context_embeding=768)
        # model_arch = UNet_Tranformer_ResBlock_Arch_Embeding(None,in_channels=3,context_embeding=768)
        states = torch.load(os.path.join(text_img_model_path, "ckpt.pth"))
        model_arch.load_state_dict(states[0])
        model_arch.to(self.device)
        model = Diffusion(image_shape=image_shape)
        model_clip, preprocess = load_from_name("ViT-B-16", device=self.device, download_root='./')
        model_clip.eval()
        self.model = model
        self.model_arch = model_arch
        self.model_clip = model_clip

    def predict(self,
                drection=Direction.center,
                dim=512,
                factor=4,
                inpaint_iter=2, 
                prompt="米饭", 
                prompt2="delicious chinese food",
                scale=7.5, 
                ddim_steps=50,
                direction=Direction.center, 
                text_img_sampler_type="generalized",
                text_img_steps=100,output_format="gif"):
        with torch.no_grad():
            text = clip.tokenize([prompt]).to(self.device)
            pad_index = self.model_clip.tokenizer.vocab['[PAD]']
            attn_mask = text.ne(pad_index).type(self.model_clip.dtype)
            cond_label = self.model_clip.bert(text, attention_mask=attn_mask)[0].type(self.model_clip.dtype)
            x = torch.randn(
                1,
                self.model.image_shape[0],
                self.model.image_shape[1],
                self.model.image_shape[2],
                device=self.device,
            )
            print("condition text dim is :",cond_label.shape)
            img = self.model.sample_image(x,
                                         self.model_arch,
                                         cond_label.float().to(self.device),
                                         sample_type=text_img_sampler_type, 
                                         timesteps=text_img_steps)
            img = inverse_data_transform(img)
            img = img.cpu()[0].permute(1,2,0).numpy()*255
            img = Image.fromarray(np.uint8(img)).resize((dim, dim))
            if direction != Direction.center:
                inpaint_iter = 10
                img = self_inpaint2(self.inpaint_sampler, img, dim, factor, inpaint_iter, prompt2, scale, ddim_steps, direction=direction)
            else:
                img = self_inpaint(self.inpaint_sampler, img, dim, factor, inpaint_iter, prompt2, scale, ddim_steps, direction=direction)
            if direction == Direction.center:
                frames = zoom_in(np.array(img), factor=factor, n_self_paste=3, n_frames=60)
                fps=20
            else:
                print(len(img))
                frames = [np.array(i) for i in img]
                fps = 5

            output_format = str(output_format)
            ext = output_format.split(".")[-1]
            file_name = str(time.time()).split(".")[0]
            fname = f"infinit_zoom_{file_name}.{ext}"
            write_frames(frames, fname=fname, fps=fps)

            print("success!","generate at ",fname)

