
import argparse
import numpy as np
import torch as th
import torch.distributed as dist
from einops import rearrange, repeat
import sys,os
sys.path.append(os.path.dirname (os.path.dirname (os.path.abspath (__file__))))
from mm_diffusion import dist_util, logger
from mm_diffusion.multimodal_script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict
)
from mm_diffusion.script_util import (
    image_sr_model_and_diffusion_defaults,
    image_sr_create_model_and_diffusion
)
from mm_diffusion.common import set_seed_logger_random, save_audio, save_img, save_multimodal

from mm_diffusion.multimodal_dpm_solver_plus import DPM_Solver as multimodal_DPM_Solver
from mm_diffusion.dpm_solver_plus import DPM_Solver as singlemodal_DPM_Solver

def main():
    args = create_argparser().parse_args()
    args.video_size = [int(i) for i in args.video_size.split(',')]
    args.audio_size = [int(i) for i in args.audio_size.split(',')]
    
    dist_util.setup_dist(args.devices)
    logger.configure(args.output_dir)
    args = set_seed_logger_random(args)

    logger.log("creating model and diffusion...")
    multimodal_model, multimodal_diffusion = create_model_and_diffusion(
         **args_to_dict(args, [key for key in model_and_diffusion_defaults().keys()])
    )
    sr_model, sr_diffusion = image_sr_create_model_and_diffusion(
        **args_to_dict(args, [key for key in image_sr_model_and_diffusion_defaults().keys()])
    )

    if os.path.isdir(args.multimodal_model_path):
        multimodal_name_list = [
            model_name for model_name in os.listdir(args.multimodal_model_path)
            if (model_name.startswith('model') and model_name.endswith('.pt') and 
                int(model_name.split('.')[0][5:]) >= args.skip_steps)
        ]
        multimodal_name_list.sort()
        multimodal_name_list = [os.path.join(args.multimodal_model_path, model_name) for model_name in multimodal_name_list]
    else:
        multimodal_name_list = [model_path for model_path in args.multimodal_model_path.split(',')]
        
    logger.log(f"models to be processed: {multimodal_name_list}")

    if os.path.exists(args.sr_model_path):
        sr_model.load_state_dict_(
            dist_util.load_state_dict(args.sr_model_path, map_location="cpu"), is_strict=args.is_strict
        )
        sr_model.to(dist_util.dev())
        if args.use_fp16:
            sr_model.convert_to_fp16()
    else:
        sr_model = None

    sr_noise = None
    if os.path.exists(args.load_noise):
        sr_noise = np.load(args.load_noise)
        sr_noise = th.tensor(sr_noise).to(dist_util.dev()).unsqueeze(0)
        sr_noise = repeat(sr_noise, 'b c h w -> (b repeat) c h w', repeat=args.batch_size * args.video_size[0])
        if dist.get_rank() == 0:
            logger.log(f"Loaded noise from {args.load_noise}")

    for model_path in multimodal_name_list:
        multimodal_model.load_state_dict_(
            dist_util.load_state_dict(model_path, map_location="cpu"), is_strict=args.is_strict
        )
        
        multimodal_model.to(dist_util.dev())
        if args.use_fp16:
            multimodal_model.convert_to_fp16()
        multimodal_model.eval()

        logger.log(f"Generating samples for {model_path}")
        model_name = model_path.split('/')[-1]

        groups = 0
        multimodal_save_path = os.path.join(args.output_dir, model_name, 'original')
        sr_save_path = os.path.join(args.output_dir, model_name, 'sr_mp4')
        audio_save_path = os.path.join(args.output_dir, model_name, 'audios')
        img_save_path = os.path.join(args.output_dir, model_name, 'img')
        if dist.get_rank() == 0:
            os.makedirs(multimodal_save_path, exist_ok=True)
            os.makedirs(sr_save_path, exist_ok=True)
            os.makedirs(audio_save_path, exist_ok=True)
            os.makedirs(img_save_path, exist_ok=True)

        while groups * args.batch_size * dist.get_world_size() < args.all_save_num: 
            model_kwargs = {}
            if args.class_cond:
                classes = th.randint(
                   low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes

            shape = {
                "video": (args.batch_size, *args.video_size),
                "audio": (args.batch_size, *args.audio_size)
            }

            if args.sample_fn == 'dpm_solver':
                dpm_solver = multimodal_DPM_Solver(
                    model=multimodal_model,
                    alphas_cumprod=th.tensor(multimodal_diffusion.alphas_cumprod, dtype=th.float32)
                )
                x_T = {
                    "video": th.randn(shape["video"]).to(dist_util.dev()),
                    "audio": th.randn(shape["audio"]).to(dist_util.dev())
                }
                sample = dpm_solver.sample(
                    x_T,
                    steps=20,
                    order=3,
                    skip_type="logSNR",
                    method="singlestep",
                )
            elif args.sample_fn == 'dpm_solver++':
                dpm_solver = multimodal_DPM_Solver(
                    model=multimodal_model,
                    alphas_cumprod=th.tensor(multimodal_diffusion.alphas_cumprod, dtype=th.float32),
                    predict_x0=True,
                    thresholding=True
                )
                x_T = {
                    "video": th.randn(shape["video"]).to(dist_util.dev()),
                    "audio": th.randn(shape["audio"]).to(dist_util.dev())
                }
                sample = dpm_solver.sample(
                    x_T,
                    steps=20,
                    order=2,
                    skip_type="logSNR",
                    method="adaptive",
                )
            else:
                sample_fn = (
                    multimodal_diffusion.p_sample_loop if args.sample_fn == "ddpm" else multimodal_diffusion.ddim_sample_loop
                )
                sample = sample_fn(
                    multimodal_model,
                    shape=shape,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )

            video = ((sample["video"] + 1) * 127.5).clamp(0, 255).to(th.uint8)
            audio = sample["audio"]              
            video = video.permute(0, 1, 3, 4, 2).contiguous()

            all_videos = video.cpu().numpy()
            all_audios = audio.cpu().numpy()

            idx = 0
            for video, audio in zip(all_videos, all_audios):
                video_output_path = os.path.join(multimodal_save_path, f"{args.sample_fn}_samples_{groups}_{dist.get_rank()}_{idx}.{args.save_type}")
                audio_output_path = os.path.join(audio_save_path, f"{args.sample_fn}_samples_{groups}_{dist.get_rank()}_{idx}.wav")
                img_output_path = os.path.join(img_save_path, f"{args.sample_fn}_samples_{groups}_{dist.get_rank()}_{idx}")
                save_multimodal(video, audio, video_output_path, args)
                save_audio(audio, audio_output_path, args.audio_fps)
                save_img(video, img_output_path)
                idx += 1                    

            groups += 1
            dist.barrier()

    logger.log("Sampling complete")

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=16,
        sample_fn="dpm_solver",
        sr_model_path="",
        multimodal_model_path="",
        output_dir="",
        save_type="mp4",
        classifier_scale=0,
        devices=None,
        is_strict=True,
        all_save_num=1024,
        seed=42,
        video_fps=10,
        audio_fps=16000,
        load_noise=""
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(image_sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
