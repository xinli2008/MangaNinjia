import argparse
import logging
import os
import random
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
import torch.nn as nn
from inference.manganinjia_pipeline import MangaNinjiaPipeline
from diffusers import (
    ControlNetModel,
    DiffusionPipeline,
    DDIMScheduler,
    AutoencoderKL,
)
from src.models.mutual_self_attention_multi_scale import ReferenceAttentionControl
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.refunet_2d_condition import RefUNet2DConditionModel
from src.point_network import PointNet
from src.annotator.lineart import BatchLineartDetector

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image MangaNinjia"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=50,  # quantitative evaluation uses 50 steps
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )

    # resolution setting
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path", type=str, required=True, help="Path to original controlnet."
    )
    parser.add_argument(
        "--annotator_ckpts_path", type=str, required=True, help="Path to depth inpainting model."
    )
    parser.add_argument(
        "--manga_reference_unet_path", type=str, required=True, help="Path to depth inpainting model."
    )
    parser.add_argument(
        "--manga_main_model_path", type=str, required=True, help="Path to depth inpainting model."
    )
    parser.add_argument(
        "--manga_controlnet_model_path", type=str, required=True, help="Path to depth inpainting model."
    )
    parser.add_argument(
        "--point_net_path", type=str, required=True, help="Path to depth inpainting model."
    )
    parser.add_argument(
        "--input_reference_paths",
        nargs='+',
        default=None,
        help="input_image_paths",
    )
    parser.add_argument(
        "--input_lineart_paths",
        nargs='+',
        default=None,
        help="lineart_paths",
    )
    parser.add_argument(
        "--point_ref_paths",
        type=str,
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--point_lineart_paths",
        type=str,
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--is_lineart",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--guidance_scale_ref",
        type=float,
        default=1e-4,
        help="guidance scale for reference image",
    )
    parser.add_argument(
        "--guidance_scale_point",
        type=float,
        default=1e-4,
        help="guidance scale for points",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    denoise_steps = args.denoise_steps
    seed = args.seed
    is_lineart = args.is_lineart
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")
    if args.input_reference_paths is not None:
        assert len(args.input_reference_paths) == len(args.input_lineart_paths)
    input_reference_paths = args.input_reference_paths
    input_lineart_paths = args.input_lineart_paths
    if args.point_ref_paths is not None:
        point_ref_paths = args.point_ref_paths
        point_lineart_paths = args.point_lineart_paths
        assert len(point_ref_paths) == len(point_lineart_paths)
    print(f"arguments: {args}")
    if seed is None:
        import time

        seed = int(time.time())
    generator = torch.cuda.manual_seed(seed)
    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Model --------------------
    preprocessor = BatchLineartDetector(args.annotator_ckpts_path)
    preprocessor.to(device,dtype=torch.float32) 
    in_channels_reference_unet = 4
    in_channels_denoising_unet = 4
    in_channels_controlnet = 4
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path,subfolder='scheduler')
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder='vae'
    )

    denoising_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,subfolder="unet",
        in_channels=in_channels_denoising_unet,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True
    )
            
    reference_unet = RefUNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,subfolder="unet",
        in_channels=in_channels_reference_unet,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True
    )
    refnet_tokenizer = CLIPTokenizer.from_pretrained(args.image_encoder_path)
    refnet_text_encoder = CLIPTextModel.from_pretrained(args.image_encoder_path)
    refnet_image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
        
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_model_name_or_path,
        in_channels=in_channels_controlnet,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True
    )
    controlnet_tokenizer = CLIPTokenizer.from_pretrained(args.image_encoder_path)
    controlnet_text_encoder = CLIPTextModel.from_pretrained(args.image_encoder_path)
    controlnet_image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    
    point_net=PointNet()

    controlnet.load_state_dict(
            torch.load(args.manga_controlnet_model_path, map_location="cpu"),
            strict=False,
            )
    point_net.load_state_dict(
            torch.load(args.point_net_path, map_location="cpu"),
            strict=False,
            )
    reference_unet.load_state_dict(
            torch.load(args.manga_reference_unet_path, map_location="cpu"),
            strict=False,
            )
    denoising_unet.load_state_dict(
            torch.load(args.manga_main_model_path, map_location="cpu"),
            strict=False,
            )
    pipe = MangaNinjiaPipeline(
            reference_unet=reference_unet,
            controlnet=controlnet,
            denoising_unet=denoising_unet,  
            vae=vae,
            refnet_tokenizer=refnet_tokenizer,
            refnet_text_encoder=refnet_text_encoder,
            refnet_image_encoder=refnet_image_encoder,
            controlnet_tokenizer=controlnet_tokenizer,
            controlnet_text_encoder=controlnet_text_encoder,
            controlnet_image_encoder=controlnet_image_encoder,
            scheduler=noise_scheduler,
            point_net=point_net
        )
    pipe = pipe.to(torch.device(device))

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for i in range(len(input_reference_paths)):
            input_reference_path = input_reference_paths[i]
            input_lineart_path = input_lineart_paths[i]

            # save path
            rgb_name_base = os.path.splitext(os.path.basename(input_reference_path))[0]
            pred_name_base = rgb_name_base + "_colorized"
            lineart_name_base = rgb_name_base + "_lineart"
            colored_save_path = os.path.join(
                output_dir, f"{pred_name_base}.png"
            )
            lineart_save_path = os.path.join(
                output_dir, f"{lineart_name_base}.png"
            )
            if point_ref_paths is not None:
                point_ref_path = point_ref_paths[i]
                point_lineart_path = point_lineart_paths[i]
                point_ref = torch.from_numpy(np.load(point_ref_path)).unsqueeze(0).unsqueeze(0)
                point_main = torch.from_numpy(np.load(point_lineart_path)).unsqueeze(0).unsqueeze(0)
            else:
                matrix1 = np.zeros((512, 512), dtype=np.uint8)
                matrix2 = np.zeros((512, 512), dtype=np.uint8)
                point_ref = torch.from_numpy(matrix1).unsqueeze(0).unsqueeze(0)
                point_main = torch.from_numpy(matrix2).unsqueeze(0).unsqueeze(0)
            ref_image = Image.open(input_reference_path)
            ref_image = ref_image.resize((512, 512))
            target_image = Image.open(input_lineart_path)
            target_image = target_image.resize((512, 512))
            pipe_out = pipe(
                is_lineart,
                ref_image,
                target_image,
                target_image,
                denosing_steps=denoise_steps,
                processing_res=512,
                match_input_res=True,
                batch_size=1,
                show_progress_bar=True,
                guidance_scale_ref=args.guidance_scale_ref,
                guidance_scale_point=args.guidance_scale_point,
                preprocessor=preprocessor,
                generator=generator,
                point_ref=point_ref,  
                point_main=point_main,  
            )

            if os.path.exists(colored_save_path):
                logging.warning(f"Existing file: '{colored_save_path}' will be overwritten")
            image = pipe_out.img_pil
            lineart = pipe_out.to_save_dict['edge2_black']
            image.save(colored_save_path)
            lineart.save(lineart_save_path)