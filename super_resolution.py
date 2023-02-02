import gc
import os.path

import safetensors.torch
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat, rearrange
from pytorch_lightning import seed_everything

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentUpscaleDiffusion, LatentUpscaleFinetuneDiffusion
from ldm.util import exists, instantiate_from_config

from modules import shared, modelloader, sd_hijack

torch.set_grad_enabled(False)


def initialize_model():
    model_dir = os.path.join(shared.models_path, "Upscale")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_name = os.path.join(model_dir, "x4-upscaler-ema.ckpt")

    model_url = "https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/resolve/main/x4-upscaler-ema.ckpt"

    model_dir = os.path.join(shared.models_path, "Upscaler")
    model_file = modelloader.load_models(model_dir, model_url, None, '.ckpt', model_name)
    model_file = model_file[0]
    model_config = os.path.join(shared.script_path, "extensions", "sd_smartprocess", "configs", "upscaler.yaml")

    config = OmegaConf.load(model_config)
    model = instantiate_from_config(config.model)
    model_model = torch.load(model_file)
    model.load_state_dict(model_model["state_dict"], strict=False)
    model.half()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sd_hijack.model_hijack.hijack(model)  # apply optimization

    if shared.cmd_opts.opt_channelslast:
        model = model.to(memory_format=torch.channels_last)

    model.eval()
    sampler = DDIMSampler(model)
    return sampler, model


def make_batch_sd(
        image,
        txt,
        device,
        num_samples=1,
):
    image = np.array(image.convert("RGB"))
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    batch = {
        "lr": rearrange(image, 'h w c -> 1 c h w'),
        "txt": num_samples * [txt],
    }
    batch["lr"] = repeat(batch["lr"].to(device=device),
                         "1 ... -> n ...", n=num_samples)
    return batch


def make_noise_augmentation(model, batch, noise_level=None):
    x_low = batch[model.low_scale_key]
    x_low = x_low.to(memory_format=torch.contiguous_format).float()
    x_aug, noise_level = model.low_scale_model(x_low, noise_level)
    return x_aug, noise_level


def paint(sampler, image, prompt, seed, scale, h, w, steps, num_samples=1, callback=None, eta=0., noise_level=None):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model
    seed_everything(seed)
    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, model.channels, h, w)
    start_code = torch.from_numpy(start_code).to(
        device=device, dtype=torch.float32)

    with torch.no_grad(),\
            torch.autocast("cuda"):
        batch = make_batch_sd(image, txt=prompt, device=device, num_samples=num_samples)
        print(f"Batch: {batch}")
        c = model.cond_stage_model.encode(batch["txt"])
        c_cat = list()
        if isinstance(model, LatentUpscaleFinetuneDiffusion):
            for ck in model.concat_keys:
                cc = batch[ck]
                if exists(model.reshuffle_patch_size):
                    assert isinstance(model.reshuffle_patch_size, int)
                    cc = rearrange(cc, 'b c (p1 h) (p2 w) -> b (p1 p2 c) h w',
                                   p1=model.reshuffle_patch_size, p2=model.reshuffle_patch_size)
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)
            # cond
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}
            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
        elif isinstance(model, LatentUpscaleDiffusion):
            x_augment, noise_level = make_noise_augmentation(
                model, batch, noise_level)
            cond = {"c_concat": [x_augment],
                    "c_crossattn": [c], "c_adm": noise_level}
            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [x_augment], "c_crossattn": [
                uc_cross], "c_adm": noise_level}
        else:
            raise NotImplementedError()

        shape = [model.channels, h, w]
        samples, intermediates = sampler.sample(
            steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            x_T=start_code,
            callback=callback
        )
    with torch.no_grad():
        x_samples_ddim = model.decode_first_stage(samples)
    result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    return [Image.fromarray(img.astype(np.uint8)) for img in result]


def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded


def predict(sampler, input_image, prompt, steps, num_samples, scale, seed, eta, noise_level):
    init_image = input_image.convert("RGB")
    image = pad_image(init_image)  # resize to integer multiple of 32
    width, height = image.size

    noise_level = torch.Tensor(
        num_samples * [noise_level]).to(sampler.model.device).long()
    sampler.make_schedule(steps, ddim_eta=eta, verbose=True)
    result = paint(
        sampler=sampler,
        image=image,
        prompt=prompt,
        seed=seed,
        scale=scale,
        h=height, w=width, steps=steps,
        num_samples=num_samples,
        callback=None,
        noise_level=noise_level
    )
    return result

def super_resolution(self, images, steps=50, target_scale=2, half_attention=False):
    model = self.load_model_from_config(half_attention)
    gc.collect()
    if torch.cuda.is_available:
        torch.cuda.empty_cache()

    # Run settings
    diffusion_steps = int(steps)
    eta = 1.0
    output = []
    for image in images:
        im_og = image
        width_og, height_og = im_og.size
        # If we can adjust the max upscale size, then the 4 below should be our variable
        down_sample_rate = target_scale / 4
        wd = width_og * down_sample_rate
        hd = height_og * down_sample_rate
        width_downsampled_pre = int(np.ceil(wd))
        height_downsampled_pre = int(np.ceil(hd))

        if down_sample_rate != 1:
            print(
                f'Downsampling from [{width_og}, {height_og}] to [{width_downsampled_pre}, {height_downsampled_pre}]')
            im_og = im_og.resize((width_downsampled_pre, height_downsampled_pre), Image.LANCZOS)
        else:
            print(f"Down sample rate is 1 from {target_scale} / 4 (Not downsampling)")

        # pad width and height to multiples of 64, pads with the edge values of image to avoid artifacts
        pad_w, pad_h = np.max(((2, 2), np.ceil(np.array(im_og.size) / 64).astype(int)), axis=0) * 64 - im_og.size
        im_padded = Image.fromarray(np.pad(np.array(im_og), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))

        logs = self.run(model["model"], im_padded, diffusion_steps, eta)

        sample = logs["sample"]
        sample = sample.detach().cpu()
        sample = torch.clamp(sample, -1., 1.)
        sample = (sample + 1.) / 2. * 255
        sample = sample.numpy().astype(np.uint8)
        sample = np.transpose(sample, (0, 2, 3, 1))
        a = Image.fromarray(sample[0])

        # remove padding
        a = a.crop((0, 0) + tuple(np.array(im_og.size) * 4))
        output.append(a)

    del model
    gc.collect()
    if torch.cuda.is_available:
        torch.cuda.empty_cache()

    return output



