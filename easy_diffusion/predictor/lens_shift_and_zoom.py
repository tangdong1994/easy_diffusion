import torch
#from torch import autocast #torch version >1.8.1
from torch.cuda.amp import autocast # version1.8.1
from math import log
import imageio
import numpy as np
from PIL import Image
from .config import Direction
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange, repeat
from ..ldm.util import instantiate_from_config
from ..ldm.models.diffusion.ddim import DDIMSampler
from ..ddim.diffusion import Diffusion



def resize_img(img, size):
    return np.array(Image.fromarray(img).resize((size, size)))

def generate_mask(dim, factor, direction):
    center = dim // 2
    scaler_factor = dim // factor
    mask_len = scaler_factor // 2
    mask = np.zeros((dim, dim), dtype=np.uint8)
    if direction.value == 1:
        # 在原图上面准备一个方形mask
        # prepare mask up on original image
        mask[:mask_len, :] = 255
        mask = Image.fromarray(mask)
    elif direction.value == 2:
        # 在原图下面准备一个长方形mask
        # prepare mask down on original image
        mask[-mask_len:, :] = 255
        mask = Image.fromarray(mask)
    elif direction.value == 3:
        # 在原图左边准备一个长方形mask
        # prepare mask left on original image
        mask[:, :mask_len] = 255
        mask = Image.fromarray(mask)
    elif direction.value == 4:
        # 在原图右边准备一个长方形mask
        # prepare mask right on original image
        mask[:, -mask_len:] = 255
        mask = Image.fromarray(mask)
    elif direction.value == 5:
        # 准备一个方环形的mask，mask里面放置的是resize过后的原图，mask外面的是原大小的图
        # Prepare mask surrounding middle
        mask[center - scaler_factor:center + scaler_factor, center - scaler_factor:center + scaler_factor] = 255
        mask[center - mask_len:center + mask_len, center - mask_len:center + mask_len] = 0
        mask = Image.fromarray(mask)
    elif direction.value == 6:
        # 在原图左上位置准备方形mask
        # prepare mask left_up on original image
        mask[:mask_len,:]=255
        mask[:,:mask_len]=255
        mask = Image.fromarray(mask)
    elif direction.value == 7:
        # 在原图左下位置准备方形mask
        # prepare mask left_up on original image
        mask[-mask_len:,:]=255
        mask[:,:mask_len]=255
        mask = Image.fromarray(mask)
    elif direction.value == 8:
        # 在原图右上位置准备方形mask
        # prepare mask right_up on original image
        mask[:mask_len,:]=255
        mask[:,-mask_len:]=255
        mask = Image.fromarray(mask)
    elif direction.value == 9:
        # 在原图右下位置准备方形mask
        # prepare mask right_down on original image
        mask[-mask_len:,:]=255
        mask[:,-mask_len:]=255
        mask = Image.fromarray(mask)
    elif direction.value == 10:
        mask[-mask_len:,:]=255
        mask[:mask_len,:]=255
        mask[:,-mask_len:]=255
        mask[:,:mask_len]=255
        mask = Image.fromarray(mask)
    return mask

def process_image(dim, factor, image, direction):
    center = dim // 2
    scaler_factor = dim // factor
    mask_len = scaler_factor // 2
    if direction.value == 1:
        # 将原图往下移动
        # move original image down
        image = np.array(image)
        image[mask_len:,:] = image[:-mask_len,:]
        
    elif direction.value == 2:
        # 在原图往上偏移
        # move original image up
        image = np.array(image)
        image[:-mask_len,:] = image[mask_len:,:]
        
    elif direction.value == 3:
        # 将原图往右偏移
        # move original image right
        image = np.array(image)
        image[:,mask_len:] = image[:,:-mask_len]
        
    elif direction.value == 4:
        # 将原图往左偏移
        # move original image right
        image = np.array(image)
        image[:,:-mask_len] = image[:,mask_len:]
    elif direction.value == 5:
        # resize original image
        # 将原图resize到一定大小
        downscaled_image = np.array(image.resize((scaler_factor, scaler_factor)))
        image = np.array(image)
        image[center - mask_len:center + mask_len, center - mask_len:center + mask_len] = downscaled_image
    elif direction.value == 6:
        # 原图往右下移动
        # move original image right down
        image = np.array(image)
        image[mask_len:,:] = image[:-mask_len,:]
        image[:,mask_len:] = image[:,:-mask_len]
    elif direction.value == 7:
        # 原图往右上移动
        # move original image right up
        image = np.array(image)
        image[:-mask_len,:] = image[mask_len:,:]
        image[:,mask_len:] = image[:,:-mask_len]
    elif direction.value == 8:
        # 原图往左下移动
        # move original image right up
        image = np.array(image)
        image[mask_len:,:] = image[:-mask_len,:]
        image[:,:-mask_len] = image[:,mask_len:]
    elif direction.value == 9:
        # 在原图往左上移动
        # move original image right up
        image = np.array(image)
        image[:-mask_len,:] = image[mask_len:,:]
        image[:,:-mask_len] = image[:,mask_len:]
    elif direction.value == 10:
        # resize original image
        # 将原图resize到一定大小
        downscaled_image = np.array(image.resize((dim - scaler_factor, dim - scaler_factor)))
        image = np.array(image)
        image[mask_len:dim - mask_len, mask_len:dim - mask_len] = downscaled_image
    return image

def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch


def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

    # with autocast("cuda"): torch version >1.8.1
    with autocast():
        batch = make_batch_sd(image, mask, txt=prompt, device=device, num_samples=num_samples)

        c = model.cond_stage_model.encode(batch["txt"])

        c_cat = list()
        for ck in model.concat_keys:
            cc = batch[ck].float()
            if ck != model.masked_image_key:
                bchw = [num_samples, 4, h // 8, w // 8]
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            else:
                cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)

        # cond
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}

        # uncond cond
        uc_cross = model.get_unconditional_conditioning(num_samples, "")
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

        shape = [model.channels, h // 8, w // 8]
        samples_cfg, intermediates = sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=1.0,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            x_T=start_code,
        )
        x_samples_ddim = model.decode_first_stage(samples_cfg)

        result = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                             min=0.0, max=1.0)

        result = result.cpu().numpy().transpose(0, 2, 3, 1)
        # result, has_nsfw_concept = check_safety(result)
        result = result * 255

    result = [Image.fromarray(img.astype(np.uint8)) for img in result]
    # result = [put_watermark(img) for img in result]
    return result

def self_inpaint(inpaint_sampler, image, dim, factor, inpaint_iter, prompt, scale, ddim_steps, direction=Direction.center):
    with torch.no_grad():
        mask = generate_mask(dim, factor, direction)
        image.save(f"debug-0.png")
        mask.save(f"debug-mask-0.png")
        for i in range(inpaint_iter):
            image = process_image(dim, factor, image, direction)
            image = Image.fromarray(image)

            image = inpaint(
                sampler=inpaint_sampler,
                image=image,
                mask=mask,
                prompt=prompt,
                seed=0,
                scale=scale,
                ddim_steps=ddim_steps,
                num_samples=1,
                h=dim, w=dim
            )[0]
            image.save(f"debug-{i + 1}.png")

        return image
    
def self_inpaint2(inpaint_sampler, image, dim, factor, inpaint_iter, prompt, scale, ddim_steps, direction=Direction.center):
    with torch.no_grad():
        mask = generate_mask(dim, factor, direction)
        image.save(f"debug-0.png")
        mask.save(f"debug-mask-0.png")
        img_list = [image]
        for i in range(inpaint_iter):
            image = process_image(dim, factor, image, direction)
            image = Image.fromarray(image)

            image = inpaint(
                sampler=inpaint_sampler,
                image=image,
                mask=mask,
                prompt=prompt,
                seed=0,
                scale=scale,
                ddim_steps=ddim_steps,
                num_samples=1,
                h=dim, w=dim
            )[0]
            image.save(f"debug-{i + 1}.png")
            img_list.append(image)

        return img_list

def zoom_in(img, factor, n_self_paste=0, n_frames=120):
    frames = []
    size = img.shape[0]
    middle = size // 2
    center_size = size / factor

    # Paste the image inside itself in decreasing sizes
    for small_size in [size // factor ** (i + 1) for i in range(n_self_paste)]:
        hs = small_size // 2
        img[middle - hs:middle + hs, middle - hs:middle + hs] = resize_img(img, small_size)

    # Compute a serie of scale up factor that dictate a steady zoom-in speed
    log_base = factor ** (-1 / n_frames)
    scale_factors = [1] + [factor * log_base ** i for i in range(int(log(1 / factor, log_base)), -1, -1)]

    for scale_factor in scale_factors:
        # zoom in
        new_size = int(size * scale_factor)
        new_image = resize_img(img, new_size)
        slc = slice(new_size // 2 - size // 2, new_size // 2 + size // 2)
        new_image = new_image[slc, slc]

        # Compute center position and paste it in higher rsolution
        center_new_half_size = int(center_size * scale_factor / 2)
        slc = slice(middle - center_new_half_size, middle + center_new_half_size)
        x = new_image[slc, slc].shape[0]
        new_image[slc, slc] = resize_img(img, x)

        frame = resize_img(new_image, size)
        frames.append(frame)

    return frames

def write_frames(frames, fname, fps=30):
    if fname.endswith('gif'):
        imageio.mimsave(fname, [Image.fromarray(frame) for frame in frames], duration=1 / fps)
    elif fname.endswith('mp4'):
        os.makedirs("tmp", exist_ok=True)
        for i, frame in enumerate(frames):
            cv2.imwrite(f"tmp/{str(i).zfill(5)}.png", frame)

        cmd = [
            "ffmpeg", "-y", "-vcodec", "png",
            "-r", str(fps),
            "-start_number", str(0),
            "-i", f"tmp/%05d.png",
            "-c:v", "libx264", "-vf", f"fps={fps}", "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "veryfast", fname,
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(stderr)
            raise RuntimeError(stderr)
    else:
        raise ValueError("Can only produce mp4 or gif files")