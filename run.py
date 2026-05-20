# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: run.py

"""
# video
 python run.py \
 --src_image assets/examples/driving/d13.mp4 \
 --dri_video assets/examples/driving/d11.mp4 \
 --cfg configs/mlx_infer.yaml \
 --paste_back \
 --animal
# pkl
 python run.py \
 --src_image assets/examples/source/s12.jpg \
 --dri_video ./results/2024-09-13-081710/d0.mp4.pkl \
 --cfg configs/mlx_infer.yaml \
 --paste_back \
 --animal
"""
import os
import argparse
import subprocess
import cv2
import time
import numpy as np
import datetime
import platform
import pickle
from omegaconf import OmegaConf
from tqdm import tqdm
from colorama import Fore, Style
from src.utils.mlx_profiles import MLX_PROFILE_CHOICES, apply_mlx_profile, describe_mlx_profile

if platform.system().lower() == 'windows':
    FFMPEG = "third_party/ffmpeg-7.0.1-full_build/bin/ffmpeg.exe"
else:
    FFMPEG = "ffmpeg"


def apply_quality_options(infer_cfg, args):
    if getattr(args, "driving_option", None):
        infer_cfg.infer_params.driving_option = args.driving_option
    det_thresh = getattr(args, "det_thresh", None)
    if det_thresh is not None:
        for group_name in ("models", "animal_models"):
            group = getattr(infer_cfg, group_name, None)
            if group is not None and "face_analysis" in group:
                group.face_analysis.det_thresh = det_thresh


def maybe_enable_auto_driving_crop(infer_cfg, vcap):
    if not infer_cfg.infer_params.get("flag_auto_crop_driving_video", False):
        return
    if infer_cfg.infer_params.flag_crop_driving_video:
        return
    width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width > 0 and height > 0 and width != height:
        infer_cfg.infer_params.flag_crop_driving_video = True
        print(f"auto enabled driving crop for non-square input: {width}x{height}")


def run_with_video(args):
    from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
    from src.utils.utils import video_has_audio

    print(Fore.RED+'Render,  Q > exit,  S > Stitching,  Z > RelativeMotion,  X > AnimationRegion,  C > CropDrivingVideo, KL > AdjustSourceScale, NM > AdjustDriverScale,  Space > Webcamassource,  R > SwitchRealtimeWebcamUpdate'+Style.RESET_ALL)
    infer_cfg = OmegaConf.load(args.cfg)
    infer_cfg.infer_params.flag_pasteback = args.paste_back
    apply_quality_options(infer_cfg, args)

    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=args.animal)
    ret = pipe.prepare_source(args.src_image, realtime=args.realtime)
    if not ret:
        print(f"no face in {args.src_image}! exit!")
        exit(1)
    if not args.dri_video or not os.path.exists(args.dri_video):
        # read frame from camera if no driving video input
        vcap = cv2.VideoCapture(0)
        if not vcap.isOpened():
            print("no camera found! exit!")
            exit(1)
    else:
        vcap = cv2.VideoCapture(args.dri_video)
    maybe_enable_auto_driving_crop(infer_cfg, vcap)
    fps = int(vcap.get(cv2.CAP_PROP_FPS))
    h, w = pipe.src_imgs[0].shape[:2]
    save_dir = f"./results/{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    # render output video
    if not args.realtime:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vsave_crop_path = os.path.join(save_dir,
                                       f"{os.path.basename(args.src_image)}-{os.path.basename(args.dri_video)}-crop.mp4")
        vout_crop = cv2.VideoWriter(vsave_crop_path, fourcc, fps, (512 * 2, 512))
        vsave_org_path = os.path.join(save_dir,
                                      f"{os.path.basename(args.src_image)}-{os.path.basename(args.dri_video)}-org.mp4")
        vout_org = cv2.VideoWriter(vsave_org_path, fourcc, fps, (w, h))

    infer_times = []
    motion_lst = []
    c_eyes_lst = []
    c_lip_lst = []

    frame_ind = 0
    while vcap.isOpened():
        ret, frame = vcap.read()
        if not ret:
            break
        t0 = time.time()
        first_frame = frame_ind == 0
        dri_crop, out_crop, out_org, dri_motion_info = pipe.run(frame, pipe.src_imgs[0], pipe.src_infos[0],
                                                                first_frame=first_frame, realtime=args.realtime)
        frame_ind += 1
        if out_crop is None:
            print(f"no face in driving frame:{frame_ind}")
            continue

        motion_lst.append(dri_motion_info[0])
        c_eyes_lst.append(dri_motion_info[1])
        c_lip_lst.append(dri_motion_info[2])

        infer_times.append(time.time() - t0)
        # print(time.time() - t0)
        dri_crop = cv2.resize(dri_crop, (512, 512))
        out_crop = np.concatenate([dri_crop, out_crop], axis=1)
        out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)
        if not args.realtime:
            vout_crop.write(out_crop)
            out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
            vout_org.write(out_org)
        else:
            if infer_cfg.infer_params.flag_pasteback:
                out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
                cv2.imshow('Render', out_org)
            else:
                # image show in realtime mode
                cv2.imshow('Render', out_crop)
            # 按下'q'键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    vcap.release()
    if not args.realtime:
        vout_crop.release()
        vout_org.release()
        if video_has_audio(args.dri_video):
            vsave_crop_path_new = os.path.splitext(vsave_crop_path)[0] + "-audio.mp4"
            subprocess.call(
                [FFMPEG, "-i", vsave_crop_path, "-i", args.dri_video,
                 "-b:v", "10M", "-c:v",
                 "libx264", "-map", "0:v", "-map", "1:a",
                 "-c:a", "aac",
                 "-pix_fmt", "yuv420p", vsave_crop_path_new, "-y", "-shortest"])
            vsave_org_path_new = os.path.splitext(vsave_org_path)[0] + "-audio.mp4"
            subprocess.call(
                [FFMPEG, "-i", vsave_org_path, "-i", args.dri_video,
                 "-b:v", "10M", "-c:v",
                 "libx264", "-map", "0:v", "-map", "1:a",
                 "-c:a", "aac",
                 "-pix_fmt", "yuv420p", vsave_org_path_new, "-y", "-shortest"])

            print(vsave_crop_path_new)
            print(vsave_org_path_new)
        else:
            print(vsave_crop_path)
            print(vsave_org_path)
    else:
        cv2.destroyAllWindows()

    print(
        "inference median time: {} ms/frame, mean time: {} ms/frame".format(np.median(infer_times) * 1000,
                                                                            np.mean(infer_times) * 1000))
    # save driving motion to pkl
    template_dct = {
        'n_frames': len(motion_lst),
        'output_fps': fps,
        'motion': motion_lst,
        'c_eyes_lst': c_eyes_lst,
        'c_lip_lst': c_lip_lst,
    }
    template_pkl_path = os.path.join(save_dir,
                                     f"{os.path.basename(args.dri_video)}.pkl")
    with open(template_pkl_path, "wb") as fw:
        pickle.dump(template_dct, fw)
    print(f"save driving motion pkl file at : {template_pkl_path}")


def run_with_pkl(args):
    from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
    from src.utils.utils import video_has_audio

    infer_cfg = OmegaConf.load(args.cfg)
    infer_cfg.infer_params.flag_pasteback = args.paste_back
    apply_quality_options(infer_cfg, args)

    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=args.animal)
    ret = pipe.prepare_source(args.src_image, realtime=args.realtime)
    if not ret:
        print(f"no face in {args.src_image}! exit!")
        return
    with open(args.dri_video, "rb") as fin:
        dri_motion_infos = pickle.load(fin)

    fps = int(dri_motion_infos["output_fps"])
    h, w = pipe.src_imgs[0].shape[:2]
    save_dir = f"./results/{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    # render output video
    if not args.realtime:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vsave_crop_path = os.path.join(save_dir,
                                       f"{os.path.basename(args.src_image)}-{os.path.basename(args.dri_video)}-crop.mp4")
        vout_crop = cv2.VideoWriter(vsave_crop_path, fourcc, fps, (512, 512))
        vsave_org_path = os.path.join(save_dir,
                                      f"{os.path.basename(args.src_image)}-{os.path.basename(args.dri_video)}-org.mp4")
        vout_org = cv2.VideoWriter(vsave_org_path, fourcc, fps, (w, h))

    infer_times = []
    motion_lst = dri_motion_infos["motion"]
    c_eyes_lst = dri_motion_infos["c_eyes_lst"] if "c_eyes_lst" in dri_motion_infos else dri_motion_infos[
        "c_d_eyes_lst"]
    c_lip_lst = dri_motion_infos["c_lip_lst"] if "c_lip_lst" in dri_motion_infos else dri_motion_infos["c_d_lip_lst"]

    frame_num = len(motion_lst)
    for frame_ind in tqdm(range(frame_num)):
        t0 = time.time()
        first_frame = frame_ind == 0
        dri_motion_info_ = [motion_lst[frame_ind], c_eyes_lst[frame_ind], c_lip_lst[frame_ind]]
        out_crop, out_org = pipe.run_with_pkl(dri_motion_info_, pipe.src_imgs[0], pipe.src_infos[0],
                                              first_frame=first_frame, realtime=args.realtime)
        if out_crop is None:
            print(f"no face in driving frame:{frame_ind}")
            continue

        infer_times.append(time.time() - t0)
        # print(time.time() - t0)
        out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)
        if not args.realtime:
            vout_crop.write(out_crop)
            out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
            vout_org.write(out_org)
        else:
            if infer_cfg.infer_params.flag_pasteback:
                out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
                cv2.imshow('Render,  Q > exit,  S > Stitching,  Z > RelativeMotion,  X > AnimationRegion,  C > CropDrivingVideo, KL > AdjustSourceScale, NM > AdjustDriverScale,  Space > Webcamassource,  R > SwitchRealtimeWebcamUpdate',out_org)
            else:
                # image show in realtime mode
                cv2.imshow('Render,  Q > exit,  S > Stitching,  Z > RelativeMotion,  X > AnimationRegion,  C > CropDrivingVideo, KL > AdjustSourceScale, NM > AdjustDriverScale,  Space > Webcamassource,  R > SwitchRealtimeWebcamUpdate', out_crop)
            # Press the 'q' key to exit the loop, r to switch realtime src_webcam update, spacebar to switch sourceisWebcam
            k = cv2.waitKey(1) & 0xFF
            key = chr(k).lower() if k < 128 else ""
            if key == 'q':
                break
            # Key for Interesting Params    
            if key == 's':
                infer_cfg.infer_params.flag_stitching = not infer_cfg.infer_params.flag_stitching
                print('flag_stitching:'+str(infer_cfg.infer_params.flag_stitching))
            if key == 'z':
                infer_cfg.infer_params.flag_relative_motion = not infer_cfg.infer_params.flag_relative_motion
                print('flag_relative_motion:'+str(infer_cfg.infer_params.flag_relative_motion))                
            if key == 'x':
                if infer_cfg.infer_params.animation_region == "all":
                    infer_cfg.infer_params.animation_region = "exp"
                    print('animation_region = "exp"')
                else:
                    infer_cfg.infer_params.animation_region = "all"
                    print('animation_region = "all"')
            if key == 'c':
                infer_cfg.infer_params.flag_crop_driving_video = not infer_cfg.infer_params.flag_crop_driving_video
                pipe.driving_crop_state = None
                print('flag_crop_driving_video:'+str(infer_cfg.infer_params.flag_crop_driving_video))  
            if key == 'v':
                infer_cfg.infer_params.flag_pasteback = not infer_cfg.infer_params.flag_pasteback
                print('flag_pasteback:'+str(infer_cfg.infer_params.flag_pasteback)) 
                
            if key == 'a':
                infer_cfg.infer_params.flag_normalize_lip = not infer_cfg.infer_params.flag_normalize_lip
                print('flag_normalize_lip:'+str(infer_cfg.infer_params.flag_normalize_lip))  
            if key == 'd':
                infer_cfg.infer_params.flag_source_video_eye_retargeting = not infer_cfg.infer_params.flag_source_video_eye_retargeting
                print('flag_source_video_eye_retargeting:'+str(infer_cfg.infer_params.flag_source_video_eye_retargeting))  
            if key == 'f':
                infer_cfg.infer_params.flag_video_editing_head_rotation = not infer_cfg.infer_params.flag_video_editing_head_rotation
                print('flag_video_editing_head_rotation:'+str(infer_cfg.infer_params.flag_video_editing_head_rotation))                 
            if key == 'g':
                infer_cfg.infer_params.flag_eye_retargeting = not infer_cfg.infer_params.flag_eye_retargeting
                print('flag_eye_retargeting:'+str(infer_cfg.infer_params.flag_eye_retargeting)) 
                
            if key == 'k':
                infer_cfg.crop_params.src_scale -= 0.1
                ret = pipe.prepare_source(args.src_image, realtime=args.realtime)
                print('src_scale:'+str(infer_cfg.crop_params.src_scale))                
            if key == 'l':
                infer_cfg.crop_params.src_scale += 0.1
                ret = pipe.prepare_source(args.src_image, realtime=args.realtime)
                print('src_scale:'+str(infer_cfg.crop_params.src_scale))  
            if key == 'n':
                infer_cfg.crop_params.dri_scale -= 0.1
                pipe.driving_crop_state = None
                print('dri_scale:'+str(infer_cfg.crop_params.dri_scale))                
            if key == 'm':
                infer_cfg.crop_params.dri_scale += 0.1
                pipe.driving_crop_state = None
                print('dri_scale:'+str(infer_cfg.crop_params.dri_scale))

    if not args.realtime:
        vout_crop.release()
        vout_org.release()
        if video_has_audio(args.dri_video):
            vsave_crop_path_new = os.path.splitext(vsave_crop_path)[0] + "-audio.mp4"
            subprocess.call(
                [FFMPEG, "-i", vsave_crop_path, "-i", args.dri_video,
                 "-b:v", "10M", "-c:v",
                 "libx264", "-map", "0:v", "-map", "1:a",
                 "-c:a", "aac",
                 "-pix_fmt", "yuv420p", vsave_crop_path_new, "-y", "-shortest"])
            vsave_org_path_new = os.path.splitext(vsave_org_path)[0] + "-audio.mp4"
            subprocess.call(
                [FFMPEG, "-i", vsave_org_path, "-i", args.dri_video,
                 "-b:v", "10M", "-c:v",
                 "libx264", "-map", "0:v", "-map", "1:a",
                 "-c:a", "aac",
                 "-pix_fmt", "yuv420p", vsave_org_path_new, "-y", "-shortest"])

            print(vsave_crop_path_new)
            print(vsave_org_path_new)
        else:
            print(vsave_crop_path)
            print(vsave_org_path)
    else:
        cv2.destroyAllWindows()

    print(
        "inference median time: {} ms/frame, mean time: {} ms/frame".format(np.median(infer_times) * 1000,
                                                                            np.mean(infer_times) * 1000))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Faster Live Portrait Pipeline')
    parser.add_argument('--src_image', required=False, type=str, default="assets/examples/source/s12.jpg",
                        help='source image')
    parser.add_argument('--dri_video', required=False, type=str, default="assets/examples/driving/d14.mp4",
                        help='driving video')
    parser.add_argument('--cfg', required=False, type=str, default="configs/mlx_infer.yaml", help='inference config')
    parser.add_argument('--realtime', action='store_true', help='realtime inference')
    parser.add_argument('--animal', action='store_true', help='use animal model')
    parser.add_argument('--paste_back', action='store_true', default=False, help='paste back to origin image')
    parser.add_argument('--det-thresh', type=float, default=None,
                        help='face detection threshold; lower values improve recall on webcam frames')
    parser.add_argument('--driving-option', choices=["expression-friendly", "pose-friendly"], default=None,
                        help='motion transfer mode from the official LivePortrait pipeline')
    parser.add_argument(
        '--mlx-profile',
        choices=MLX_PROFILE_CHOICES,
        default='quality',
        help='named MLX runtime profile; use "custom" to keep explicit FLP_MLX_* environment values',
    )
    args, unknown = parser.parse_known_args()

    applied = apply_mlx_profile(args.mlx_profile)
    if args.mlx_profile == "custom":
        print("MLX profile: custom (using explicit FLP_MLX_* values)")
    else:
        print(f"MLX profile: {args.mlx_profile} - {describe_mlx_profile(args.mlx_profile)}")
        print(
            "MLX profile settings: "
            f"temporal={applied['FLP_MLX_TEMPORAL_WARP_INTERVAL']}, "
            f"threshold={applied['FLP_MLX_TEMPORAL_WARP_THRESHOLD']}, "
            f"conv3d={applied['FLP_MLX_CONV3D_BACKEND']}, "
            f"warp_out={applied['FLP_MLX_WARP_OUT_BACKEND']}, "
            f"compress={applied['FLP_MLX_COMPRESS_BACKEND']}, "
            f"fused_deformation={applied['FLP_MLX_FUSED_DEFORMATION']}"
        )

    if args.dri_video.endswith(".pkl"):
        run_with_pkl(args)
    else:
        run_with_video(args)
