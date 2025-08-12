import argparse
import logging
import os
import os.path as osp
import cv2
import random

from easydict import EasyDict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, Compose
from diffusers import AutoencoderKLTemporalDecoder, FlowMatchEulerDiscreteScheduler

from gvm.pipelines.pipeline_gvm import GVMPipeline
from gvm.utils.inference_utils import VideoReader, VideoWriter, ImageSequenceReader, ImageSequenceWriter
from gvm.models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from tqdm import tqdm

def sequence_collate_fn(examples):
    rgb_values = torch.stack([example["image"] for example in examples])
    rgb_values = rgb_values.to(memory_format=torch.contiguous_format).float()
    return {'rgb_values': rgb_values, 'rgb_names': rgb_names}    

def impad_multi(img, multiple=32):

    target_h = int(np.ceil(img.shape[2] / multiple) * multiple)
    target_w = int(np.ceil(img.shape[3] / multiple) * multiple)

    pad_top = (target_h - img.shape[2]) // 2
    pad_bottom = target_h - img.shape[2] - pad_top
    pad_left = (target_w - img.shape[3]) // 2
    pad_right = target_w - img.shape[3] - pad_left

    padded = torch.zeros((img.shape[0], img.shape[1], target_h, target_w), dtype=img.dtype)
    padded[:, :, pad_top:pad_top + img.shape[2], pad_left:pad_left + img.shape[3]] = img

    return padded, (pad_top, pad_left, pad_bottom, pad_right)


def seed_all(seed: int = 0):
    """
    Set random seeds of all components.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def FB_blur_fusion_foreground_estimator(image, F, B, alpha, r=90):
    blurred_alpha = cv2.blur(alpha, (r, r))[:, :, None]

    blurred_FA = cv2.blur(F * alpha, (r, r))
    blurred_F = blurred_FA / (blurred_alpha + 1e-5)

    blurred_B1A = cv2.blur(B * (1 - alpha), (r, r))
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)
    F = blurred_F + alpha * \
        (image - alpha * blurred_F - (1 - alpha) * blurred_B)
    F = np.clip(F, 0, 1)
    return F, blurred_B


def FB_blur_fusion_foreground_estimator_1(image, alpha, r=90):
    alpha = alpha[:, :, None]
    return FB_blur_fusion_foreground_estimator(image, F=image, B=image, alpha=alpha, r=r)[0]


def FB_blur_fusion_foreground_estimator_2(image, alpha, r=90):
    alpha = alpha[:, :, None]
    F, blur_B = FB_blur_fusion_foreground_estimator(
        image, image, image, alpha, r)
    return FB_blur_fusion_foreground_estimator(image, F, blur_B, alpha, r=6)[0]

    
if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run video matte."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="matte",
        help="Inference mode.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="fp16",
        help=".",
    )
    parser.add_argument(
        "--model_base",
        type=str,
        default="data/weights",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--pretrain_type",
        type=str,
        default="dav",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--unet_base",
        type=str,
        default=None,
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--lora_base",
        type=str,
        default=None,
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default='zeros',
        choices=['gaussian', 'zeros'],
    )
    # data setting
    parser.add_argument(
        "--data_dir", type=str, required=True, help="input data directory."
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=1,
        help="Denoising steps, 1-3 steps work fine.",
    )
    parser.add_argument(
        "--num_frames_per_batch",
        type=int,
        default=32,
        help="Number of frames to infer per forward",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Number of frames to infer per forward",
    )
    parser.add_argument(
        "--decode_chunk_size",
        type=int,
        default=16,
        help="Number of frames to decode per forward",
    )
    parser.add_argument(
        "--num_interp_frames",
        type=int,
        default=16,
        help="Number of frames for inpaint inference",
    )
    parser.add_argument(
        "--num_overlap_frames",
        type=int,
        default=6,
        help="Number of frames to overlap between windows",
    )
    parser.add_argument(
        "--use_unet_interp",
        action="store_true",
        default=False,
        help="Whether use interploation unet",
    )
    parser.add_argument(
        "--use_clip_img_emb",
        action="store_true",
        default=False,
        help="Whether use interploation unet",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=720,  # decrease for faster inference and lower memory usage
        help="Maximum resolution for inference.",
    )
    parser.add_argument(
        "--max_resolution",
        type=int,
        default=1024,  # decrease for faster inference and lower memory usage
        help="Maximum resolution for inference.",
    )
    parser.add_argument(
        "--output_image_seq_only",
        action="store_true",
        default=False,
        help="Whether to disable concatenating the result with rgb image",
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    args = parser.parse_args()
    cfg = EasyDict(vars(args))

    upper_bound = 240./255.
    lower_bound = 25./ 255.

    file_name = cfg.data_dir.split("/")[-1].split(".")[0]
    is_video = cfg.data_dir.endswith(".mp4") or cfg.data_dir.endswith(".mkv") or cfg.data_dir.endswith(".gif")
    is_gif = cfg.data_dir.endswith(".gif")

    is_image_sequence = (not is_video) and (cfg.data_dir.split('.')[-1] not in ['jpg', 'png', 'jpeg', 'JPG'])
    if is_image_sequence and osp.exists(cfg.output_dir) and len(os.listdir(cfg.output_dir)) != 0:
        exit()

    if cfg.seed is None:
        import time

        cfg.seed = int(time.time())
    seed_all(cfg.seed)

    device_type = "cuda"
    device = torch.device(device_type)

    os.makedirs(cfg.output_dir, exist_ok=True)
    logging.info(f"output dir = {cfg.output_dir}")

    vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.model_base, subfolder="vae")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.model_base, 
        subfolder="scheduler"
    )
    unet_folder = cfg.unet_base if cfg.unet_base is not None else cfg.model_base

    if args.pretrain_type == 'dav':
        unet = UNetSpatioTemporalRopeConditionModel.from_pretrained(
            unet_folder, 
            subfolder="unet"
        )
    else:
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            unet_folder, 
            subfolder="unet", 
            # variant=args.variant,
            class_embed_type=None,
            # low_cpu_mem_usage=True, 
        )
        # import pdb;pdb.set_trace()

    pipe = GVMPipeline(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
    )
    # pipe.to(args.variant)
    
    if args.lora_base is not None:
        pipe.load_lora_weights(f"{args.lora_base}/pytorch_lora_weights.safetensors")
    
    pipe = pipe.to(device)

    # import pdb;pdb.set_trace()
    if is_video:
        num_interp_frames = cfg.num_interp_frames
        num_overlap_frames = cfg.num_overlap_frames
        num_frames = cfg.num_frames_per_batch

        reader = VideoReader(
            cfg.data_dir, 
            max_frames=cfg.max_frames,
            # transform=None,
            transform=Compose(
                [
                    ToTensor(),
                    Resize(size=cfg.size, max_size=cfg.max_resolution)
                ]
            )
        )
        fps = reader.frame_rate
        origin_shape = reader.origin_shape
        total_frames = len(reader)
        print('total video frames: {}'.format(total_frames))

    # elif is_image_sequence:
        # reader = ImageSequenceReader(cfg.data_dir, transform=None, dataset_name='video240k')
        # total_frames = len(reader)
        # print('total video frames: {}'.format(total_frames))
    else:
        input_root_list = file_name
        
        # raise NotImplementedError
    #     image, file_name = img_utils.read_image_sequence(cfg.data_dir)
    # else:
    #     image = img_utils.read_image(cfg.data_dir)
    # origin_shape = image[0].shape[:2]

    # origin_shape_list = [img.shape[:2] for img in image]
    # image = img_utils.imresize_max(image, cfg.max_resolution)
    # image = img_utils.imcrop_multi(image)
    # image, pad_info_list = img_utils.impad_multi(image)
    if cfg.output_image_seq_only:
        writer_alpha = ImageSequenceWriter(cfg.output_dir)

    else:
        writer_alpha = VideoWriter('{}/{}'.format(cfg.output_dir, f"{file_name}.mp4"), frame_rate=fps)
        writer_green = VideoWriter('{}/{}'.format(cfg.output_dir, f"{file_name}_green.mp4"), frame_rate=fps)
        # writer_green_1 = VideoWriter('{}/{}'.format(cfg.output_dir, f"{file_name}_green_use_pred_fg.mp4"), frame_rate=int(fps))
        # writer_fg = VideoWriter('{}/{}'.format(cfg.output_dir, f"{file_name}_fg.mp4"), frame_rate=fps)
        writer_green_seq = ImageSequenceWriter(cfg.output_dir)
        writer_alpha_seq = ImageSequenceWriter(cfg.output_dir)
        # writer_fg_seq = ImageSequenceWriter(cfg.output_dir)

    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.float16):
         # RGB tensor normalized to 0 ~ 1.
        if is_video:
            dataloader = DataLoader(reader, batch_size=cfg.num_frames_per_batch)
        else:
            reader = ImageSequenceReader(
            input_root, 
            transform=Compose(
                [
                    ToTensor(),
                    Resize(size=cfg.size, max_size=cfg.max_resolution)
                ])
            )

            # dataloader = DataLoader(
            #     reader, 
            #     batch_size=cfg.num_frames_per_batch,
            #     collate_fn=sequence_collate_fn,
            # )

        for batch_id, batch in tqdm(enumerate(dataloader)):
            # src, filename = batch
            # filename = filename[0]
            # import pdb;pdb.set_trace()
            if is_video:
                b, _, h, w = batch.shape
                filenames = []
                filenames_green = []
                for i in range(0, b):
                    file_id = batch_id * b + i
                    filenames.append("{}.jpg".format(file_id))
                    filenames_green.append("{}_comp.jpg".format(file_id))
            else:
                filenames = batch['rgb_names'] 
                batch = batch['rgb_values']
            
            # origin_shape_list = [img.shape[:2] for img in batch]
            # batch = img_utils.imresize_max(batch, cfg.max_resolution)
            batch, pad_info = impad_multi(batch)

            pipe_out = pipe(
                batch.to(device),
                # num_frames=num_frames,
                num_frames=args.num_frames_per_batch,
                num_overlap_frames=cfg.num_overlap_frames,
                num_interp_frames=cfg.num_interp_frames,
                decode_chunk_size=cfg.decode_chunk_size,
                num_inference_steps=cfg.denoise_steps,
                # mode='matte'
                mode=args.mode,
                use_clip_img_emb=args.use_clip_img_emb,
                noise_type=args.noise_type,
            )
            image = pipe_out.image
            alpha = pipe_out.alpha

            # alpha = np.repeat(alpha[...,None], 3, axis=-1)
            # crop and resize to the origin shape
            out_h, out_w = image.shape[2:]
            pad_t, pad_l, pad_b, pad_r = pad_info
            image = image[:, :, pad_t:(out_h-pad_b), pad_l:(out_w-pad_r)] # N, 3, H, W
            alpha = alpha[:, :, pad_t:(out_h-pad_b), pad_l:(out_w-pad_r)] # N, 3, H, W
            
            image = F.interpolate(image, origin_shape, mode='bilinear')
            alpha = F.interpolate(alpha, origin_shape, mode='bilinear')

            alpha[alpha>=upper_bound] = 1.0
            alpha[alpha<=lower_bound] = 0.0

            if cfg.output_image_seq_only:
                pass
            else:
                writer_alpha.write(alpha)
                writer_alpha_seq.write(alpha, filenames=filenames)