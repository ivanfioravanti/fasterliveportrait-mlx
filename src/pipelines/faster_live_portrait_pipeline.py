# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo0611@gmail.com
# @Project : FasterLivePortrait
# @FileName: faster_live_portrait_pipeline.py

import copy
import traceback
import cv2
from tqdm import tqdm
import numpy as np

from .. import models
from ..utils.crop import crop_image, parse_bbox_from_landmark, crop_image_by_bbox, paste_back_numpy
from ..utils.utils import resize_to_limit, prepare_paste_back, get_rotation_matrix, calc_lip_close_ratio, \
    calc_eye_close_ratio, transform_keypoint, concat_feat
from ..utils.tqdm_utils import configure_tqdm_single_process
from src.utils import utils

configure_tqdm_single_process()


def _image_to_uint8_numpy(image):
    if isinstance(image, np.ndarray):
        return image if image.dtype == np.uint8 else image.astype(np.uint8)
    if hasattr(image, "detach") and hasattr(image, "cpu"):
        image = image.detach().cpu().numpy()
    else:
        image = np.asarray(image)
    return image if image.dtype == np.uint8 else image.astype(np.uint8)


class FasterLivePortraitPipeline:
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.init(**kwargs)

    def init(self, **kwargs):
        self.init_vars(**kwargs)
        self.init_models(**kwargs)

    def update_cfg(self, args_user):
        update_ret = False
        for key in args_user:
            if key in self.cfg.infer_params:
                if self.cfg.infer_params[key] != args_user[key]:
                    update_ret = True
                print("update infer cfg {} from {} to {}".format(key, self.cfg.infer_params[key], args_user[key]))
                self.cfg.infer_params[key] = args_user[key]
            elif key in self.cfg.crop_params:
                if self.cfg.crop_params[key] != args_user[key]:
                    update_ret = True
                print("update crop cfg {} from {} to {}".format(key, self.cfg.crop_params[key], args_user[key]))
                self.cfg.crop_params[key] = args_user[key]
            else:
                if key in self.cfg.infer_params and self.cfg.infer_params[key] != args_user[key]:
                    update_ret = True
                print("add {}:{} to infer cfg".format(key, args_user[key]))
                self.cfg.infer_params[key] = args_user[key]
        return update_ret

    def clean_models(self, **kwargs):
        """
        clean model
        :param kwargs:
        :return:
        """
        for key in list(self.model_dict.keys()):
            del self.model_dict[key]
        self.model_dict = {}

    def init_models(self, **kwargs):
        if not kwargs.get("is_animal", False):
            print("load Human Model >>>")
            self.is_animal = False
            self.model_dict = {}
            for model_name in self.cfg.models:
                print(f"loading model: {model_name}")
                print(self.cfg.models[model_name])
                self.model_dict[model_name] = getattr(models, self.cfg.models[model_name]["name"])(
                    **self.cfg.models[model_name])
        else:
            print("load Animal Model >>>")
            self.is_animal = True
            self.model_dict = {}
            for model_name in self.cfg.animal_models:
                print(f"loading model: {model_name}")
                print(self.cfg.animal_models[model_name])
                self.model_dict[model_name] = getattr(models, self.cfg.animal_models[model_name]["name"])(
                    **self.cfg.animal_models[model_name])

    def init_vars(self, **kwargs):
        self.mask_crop = cv2.imread(self.cfg.infer_params.mask_crop_path, cv2.IMREAD_COLOR)
        self.frame_id = 0
        self.src_lmk_pre = None
        self.R_d_0 = None
        self.x_d_0_info = None
        self.driving_crop_state = None
        self.expression_motion_cache = {}
        self.R_d_smooth = utils.OneEuroFilter(4, 0.3)
        self.exp_smooth = utils.OneEuroFilter(4, 0.3)

        ## 记录source的信息
        self.source_path = None
        self.src_infos = []
        self.src_imgs = []
        self.prepare_source_error = None
        self.is_source_video = False

    def calc_combined_eye_ratio(self, c_d_eyes_i, source_lmk):
        c_s_eyes = calc_eye_close_ratio(source_lmk[None])
        c_d_eyes_i = np.array(c_d_eyes_i).reshape(1, 1)
        # [c_s,eyes, c_d,eyes,i]
        combined_eye_ratio_tensor = np.concatenate([c_s_eyes, c_d_eyes_i], axis=1)
        return combined_eye_ratio_tensor

    def calc_combined_lip_ratio(self, c_d_lip_i, source_lmk):
        c_s_lip = calc_lip_close_ratio(source_lmk[None])
        c_d_lip_i = np.array(c_d_lip_i).reshape(1, 1)  # 1x1
        # [c_s,lip, c_d,lip,i]
        combined_lip_ratio_tensor = np.concatenate([c_s_lip, c_d_lip_i], axis=1)  # 1x2
        return combined_lip_ratio_tensor

    def stabilize_driving_bbox(self, bbox):
        if not self.cfg.infer_params.get("flag_stabilize_driving_crop", False):
            return bbox

        left, top, right, bot = [float(x) for x in bbox]
        center = np.array([(left + right) * 0.5, (top + bot) * 0.5], dtype=np.float32)
        size = float(max(right - left, bot - top))
        if size <= 1:
            return bbox

        if self.driving_crop_state is None:
            self.driving_crop_state = {"center": center, "size": size}
        else:
            center_alpha = float(self.cfg.infer_params.get("driving_crop_center_smoothing", 0.85))
            scale_alpha = float(self.cfg.infer_params.get("driving_crop_scale_smoothing", 0.98))
            center_alpha = min(max(center_alpha, 0.0), 0.99)
            scale_alpha = min(max(scale_alpha, 0.0), 0.99)
            prev_center = self.driving_crop_state["center"]
            prev_size = self.driving_crop_state["size"]
            center = prev_center * center_alpha + center * (1.0 - center_alpha)
            if self.cfg.infer_params.get("flag_lock_driving_crop_scale", True):
                size = prev_size
            else:
                size = prev_size * scale_alpha + size * (1.0 - scale_alpha)
            self.driving_crop_state = {"center": center, "size": size}

        half = self.driving_crop_state["size"] * 0.5
        center = self.driving_crop_state["center"]
        return [
            float(center[0] - half),
            float(center[1] - half),
            float(center[0] + half),
            float(center[1] + half),
        ]

    def prepare_source(self, source_path, **kwargs):
        print(f"process source:{source_path} >>>>>>>>")
        self.prepare_source_error = None
        try:
            if utils.is_video(source_path):
                self.is_source_video = True
            else:
                self.is_source_video = False

            if self.is_source_video:
                src_imgs_bgr = []
                src_vcap = cv2.VideoCapture(source_path)
                while True:
                    ret, frame = src_vcap.read()
                    if not ret:
                        break
                    src_imgs_bgr.append(frame)
                src_vcap.release()
            else:
                img_bgr = cv2.imread(source_path, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    self.prepare_source_error = "Could not read source image."
                    return False
                src_imgs_bgr = [img_bgr]

            if not src_imgs_bgr:
                self.prepare_source_error = "Could not read any source frames."
                return False

            self.src_imgs = []
            self.src_infos = []
            self.source_path = source_path

            for ii, img_bgr in tqdm(enumerate(src_imgs_bgr), total=len(src_imgs_bgr)):
                img_bgr = resize_to_limit(img_bgr, self.cfg.infer_params.source_max_dim,
                                          self.cfg.infer_params.source_division)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                src_faces = []
                if self.is_animal:
                    src_faces = self.model_dict["face_analysis"].predict(img_bgr)
                    if len(src_faces) == 0:
                        self.prepare_source_error = (
                            "No animal face detected in the source. "
                            "Check Animal mode and use a clearer animal face image."
                        )
                        continue
                    self.src_imgs.append(img_rgb)
                else:
                    src_faces = self.model_dict["face_analysis"].predict(img_bgr)
                    if len(src_faces) == 0:
                        self.prepare_source_error = (
                            "No human face detected in the source. "
                            "If this is an animal image, enable Animal mode."
                        )
                        print("No human face detected in this image.")
                        continue
                    self.src_imgs.append(img_rgb)
                    # 如果是实时，只关注最大的那张脸
                    if kwargs.get("realtime", False):
                        src_faces = src_faces[:1]

                crop_infos = []
                for i in range(len(src_faces)):
                    # NOTE: temporarily only pick the first face, to support multiple face in the future
                    lmk = src_faces[i]
                    # crop the face
                    ret_dct = crop_image(
                        img_rgb,  # ndarray
                        lmk,  # 106x2 or Nx2
                        dsize=self.cfg.crop_params.src_dsize,
                        scale=self.cfg.crop_params.src_scale,
                        vx_ratio=self.cfg.crop_params.src_vx_ratio,
                        vy_ratio=self.cfg.crop_params.src_vy_ratio,
                    )
                    if self.is_animal:
                        ret_dct["lmk_crop"] = lmk
                    else:
                        lmk = self.model_dict["landmark"].predict(img_rgb, lmk)
                        ret_dct["lmk_crop"] = lmk
                        ret_dct["lmk_crop_256x256"] = ret_dct["lmk_crop"] * 256 / self.cfg.crop_params.src_dsize

                    # update a 256x256 version for network input
                    ret_dct["img_crop_256x256"] = cv2.resize(
                        ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA
                    )
                    crop_infos.append(ret_dct)

                src_infos = [[] for _ in range(len(crop_infos))]
                for i, crop_info in enumerate(crop_infos):
                    source_lmk = crop_info['lmk_crop']
                    img_crop_256x256 = crop_info['img_crop_256x256']
                    pitch, yaw, roll, t, exp, scale, kp = self.model_dict["motion_extractor"].predict(
                        img_crop_256x256)
                    x_s_info = {
                        "pitch": pitch,
                        "yaw": yaw,
                        "roll": roll,
                        "t": t,
                        "exp": exp,
                        "scale": scale,
                        "kp": kp
                    }
                    src_infos[i].append(copy.deepcopy(x_s_info))
                    x_c_s = kp
                    R_s = get_rotation_matrix(pitch, yaw, roll)
                    f_s = self.model_dict["app_feat_extractor"].predict(img_crop_256x256)
                    x_s = transform_keypoint(pitch, yaw, roll, t, exp, scale, kp)
                    src_infos[i].extend([source_lmk.copy(), R_s.copy(), f_s.copy(), x_s.copy(), x_c_s.copy()])
                    if not self.is_animal:
                        flag_lip_zero = self.cfg.infer_params.flag_normalize_lip  # not overwrite
                        if flag_lip_zero:
                            # let lip-open scalar to be 0 at first
                            # 似乎要调参？
                            c_d_lip_before_animation = [0.05]
                            combined_lip_ratio_tensor_before_animation = self.calc_combined_lip_ratio(
                                c_d_lip_before_animation, source_lmk.copy())
                            if combined_lip_ratio_tensor_before_animation[0][
                                0] < self.cfg.infer_params.lip_normalize_threshold:
                                flag_lip_zero = False
                                src_infos[i].append(None)
                                src_infos[i].append(flag_lip_zero)
                            else:
                                lip_delta_before_animation = self.model_dict['stitching_lip_retarget'].predict(
                                    concat_feat(x_s, combined_lip_ratio_tensor_before_animation))
                                src_infos[i].append(lip_delta_before_animation.copy())
                                src_infos[i].append(flag_lip_zero)
                        else:
                            src_infos[i].append(None)
                            src_infos[i].append(flag_lip_zero)
                    else:
                        src_infos[i].append(None)
                        src_infos[i].append(False)

                    ######## prepare for pasteback ########
                    if self.cfg.infer_params.flag_pasteback and self.cfg.infer_params.flag_do_crop:
                        mask_ori_float = prepare_paste_back(self.mask_crop, crop_info['M_c2o'],
                                                            dsize=(img_rgb.shape[1], img_rgb.shape[0]))
                        src_infos[i].append(mask_ori_float)
                    else:
                        src_infos[i].append(None)
                    src_infos[i].append(crop_info['M_c2o'])
                self.src_infos.append(src_infos[:])
            print(f"finish process source:{source_path} >>>>>>>>")
            if len(self.src_infos) == 0 and self.prepare_source_error is None:
                self.prepare_source_error = "No usable source frames were prepared."
            return len(self.src_infos) > 0
        except Exception as exc:
            self.prepare_source_error = f"Failed to process source: {exc}"
            traceback.print_exc()
            return False

    def retarget_eye(self, kp_source, eye_close_ratio):
        """
        kp_source: BxNx3
        eye_close_ratio: Bx3
        Return: Bx(3*num_kp+2)
        """
        feat_eye = concat_feat(kp_source, eye_close_ratio)
        delta = self.model_dict['stitching_eye_retarget'].predict(feat_eye)
        return delta

    def retarget_lip(self, kp_source, lip_close_ratio):
        """
        kp_source: BxNx3
        lip_close_ratio: Bx2
        """
        feat_lip = concat_feat(kp_source, lip_close_ratio)
        delta = self.model_dict['stitching_lip_retarget'].predict(feat_lip)
        return delta

    def stitching(self, kp_source, kp_driving):
        """ conduct the stitching
        kp_source: Bxnum_kpx3
        kp_driving: Bxnum_kpx3
        """

        bs, num_kp = kp_source.shape[:2]

        kp_driving_new = kp_driving.copy()

        delta = self.model_dict['stitching'].predict(concat_feat(kp_source, kp_driving_new))

        delta_exp = delta[..., :3 * num_kp].reshape(bs, num_kp, 3)  # 1x20x3
        delta_tx_ty = delta[..., 3 * num_kp:3 * num_kp + 2].reshape(bs, 1, 2)  # 1x1x2

        kp_driving_new += delta_exp
        kp_driving_new[..., :2] += delta_tx_ty

        return kp_driving_new

    def _relative_animal_keypoints(self, j, x_s_info, x_s, x_c_s, x_d_i_info, x_d_0_info, R_d_i, R_d_0, **kwargs):
        scale_new = x_s_info["scale"]
        t_i = x_d_i_info["t"].copy()
        t_0 = x_d_0_info["t"].copy()
        t_i[..., 2] = 0
        t_0[..., 2] = 0

        x_d_i = scale_new * (x_c_s @ R_d_i + x_d_i_info["exp"]) + t_i
        x_d_0 = scale_new * (x_c_s @ R_d_0 + x_d_0_info["exp"]) + t_0

        cache = self.expression_motion_cache.setdefault(j, {})
        if kwargs.get("first_frame", False) or "animal_motion_multiplier" not in cache:
            cache["animal_motion_multiplier"] = utils.calc_motion_multiplier(x_s, x_d_0)
        return (x_d_i - x_d_0) * cache["animal_motion_multiplier"] + x_s

    def _run(self, src_info, x_d_i_info, x_d_0_info, R_d_i, R_d_0, realtime, input_eye_ratio, input_lip_ratio,
             I_p_pstbk, **kwargs):
        out_crop = None
        eye_delta_before_animation = None
        for j in range(len(src_info)):
            if self.is_source_video:
                x_s_info, source_lmk, R_s, f_s, x_s, x_c_s, lip_delta_before_animation, flag_lip_zero, mask_ori_float, M = \
                    src_info[j]
                # let lip-open scalar to be 0 at first if the input is a video and flag_relative_motion
                if not (self.cfg.infer_params.flag_normalize_lip and self.cfg.infer_params.flag_relative_motion):
                    lip_delta_before_animation = None
                # let eye-open scalar to be the same as the first frame if the latter is eye-open state
                if self.cfg.infer_params.flag_source_video_eye_retargeting and source_lmk is not None:
                    combined_eye_ratio_tensor_frame_zero = utils.calc_eye_close_ratio(src_info[0][1])
                    c_d_eye_before_animation_frame_zero = [
                        [combined_eye_ratio_tensor_frame_zero[0][:2].mean()]]
                    if c_d_eye_before_animation_frame_zero[0][
                        0] < self.cfg.infer_params.source_video_eye_retargeting_threshold:
                        c_d_eye_before_animation_frame_zero = [[0.39]]
                    combined_eye_ratio_tensor_before_animation = self.calc_combined_eye_ratio(
                        c_d_eye_before_animation_frame_zero, source_lmk)
                    eye_delta_before_animation = self.retarget_eye(x_s, combined_eye_ratio_tensor_before_animation)

                if not realtime and self.cfg.infer_params.flag_pasteback and self.cfg.infer_params.flag_do_crop:
                    mask_ori_float = prepare_paste_back(self.mask_crop, M,
                                                        dsize=(self.src_imgs[0].shape[1], self.src_imgs[0].shape[0]))
            else:
                x_s_info, source_lmk, R_s, f_s, x_s, x_c_s, lip_delta_before_animation, flag_lip_zero, mask_ori_float, M = \
                    src_info[j]
            if self.is_animal and self.cfg.infer_params.flag_relative_motion:
                x_d_i_new = self._relative_animal_keypoints(
                    j,
                    x_s_info,
                    x_s,
                    x_c_s,
                    x_d_i_info,
                    x_d_0_info,
                    R_d_i,
                    R_d_0,
                    **kwargs,
                )
            elif self.cfg.infer_params.flag_relative_motion:
                if self.cfg.infer_params.animation_region in ["all", "pose"]:
                    if self.is_source_video:
                        R_new = self.R_d_smooth.process(R_d_i)
                    else:
                        R_new = (R_d_i @ np.transpose(R_d_0, (0, 2, 1))) @ R_s
                else:
                    R_new = R_s

                delta_new = x_s_info['exp'].copy()
                x_d_exp_smooth = x_d_i_info['exp'].copy()
                if self.is_source_video:
                    x_d_exp_smooth = self.exp_smooth.process(x_d_exp_smooth)
                if self.cfg.infer_params.animation_region in ["all", "exp"]:
                    if self.is_source_video:
                        for idx in [1, 2, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
                            delta_new[:, idx, :] = x_d_exp_smooth[:, idx, :]
                        delta_new[:, 3:5, 1] = x_d_exp_smooth[:, 3:5, 1]
                        delta_new[:, 5, 2] = x_d_exp_smooth[:, 5, 2]
                        delta_new[:, 8, 2] = x_d_exp_smooth[:, 8, 2]
                        delta_new[:, 9, 1:] = x_d_exp_smooth[:, 9, 1:]
                    else:
                        delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])
                elif self.cfg.infer_params.animation_region in ["lip"]:
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        if self.is_source_video:
                            delta_new[:, lip_idx, :] = x_d_exp_smooth[:, lip_idx, :]
                        else:
                            delta_new[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp']))[:,
                                                       lip_idx, :]
                elif self.cfg.infer_params.animation_region in ["eyes"]:
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        if self.is_source_video:
                            delta_new[:, eyes_idx, :] = x_d_exp_smooth[:, eyes_idx, :]
                        else:
                            delta_new[:, eyes_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp']))[:,
                                                        eyes_idx, :]
                if self.cfg.infer_params.animation_region in ["all"]:
                    if self.cfg.infer_params.get("flag_lock_driving_motion_scale", False):
                        scale_new = x_s_info['scale']
                    else:
                        scale_new = x_s_info['scale'] if self.is_source_video else x_s_info['scale'] * (
                                x_d_i_info['scale'] / x_d_0_info['scale'])
                else:
                    scale_new = x_s_info['scale']
                if self.cfg.infer_params.animation_region in ["all"]:
                    t_new = x_s_info['t'] if self.is_source_video else x_s_info['t'] + (
                            x_d_i_info['t'] - x_d_0_info['t'])
                else:
                    t_new = x_s_info['t']
            else:
                if self.cfg.infer_params.animation_region in ["all", "pose"]:
                    if self.is_source_video:
                        R_new = self.R_d_smooth.process(R_d_i)
                    else:
                        R_new = R_d_i
                else:
                    R_new = R_s

                delta_new = x_s_info['exp'].copy()
                x_d_exp_smooth = x_d_i_info['exp'].copy()
                if self.is_source_video:
                    x_d_exp_smooth = self.exp_smooth.process(x_d_exp_smooth)
                if self.cfg.infer_params.animation_region in ["all", "exp"]:
                    for idx in [1, 2, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
                        delta_new[:, idx, :] = x_d_exp_smooth[:, idx, :] if self.is_source_video else x_d_i_info['exp'][
                                                                                                      :, idx, :]
                    delta_new[:, 3:5, 1] = x_d_exp_smooth[:, 3:5, 1] if self.is_source_video else x_d_i_info['exp'][:,
                                                                                                  3:5, 1]
                    delta_new[:, 5, 2] = x_d_exp_smooth[:, 5, 2] if self.is_source_video else x_d_i_info['exp'][:,
                                                                                              5, 2]
                    delta_new[:, 8, 2] = x_d_exp_smooth[:, 8, 2] if self.is_source_video else x_d_i_info['exp'][:,
                                                                                              8, 2]
                    delta_new[:, 9, 1:] = x_d_exp_smooth[:, 9, 1:] if self.is_source_video else x_d_i_info['exp'][:,
                                                                                                9, 1:]
                elif self.cfg.infer_params.animation_region in ["lip"]:
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        delta_new[:, lip_idx, :] = x_d_exp_smooth[:, lip_idx, :] if self.is_source_video else \
                            x_d_i_info['exp'][:, lip_idx, :]
                elif self.cfg.infer_params.animation_region in ["eyes"]:
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        delta_new[:, eyes_idx, :] = x_d_exp_smooth[:, eyes_idx, :] if self.is_source_video else \
                            x_d_i_info['exp'][:, eyes_idx, :]
                scale_new = x_s_info['scale'].copy()
                if self.cfg.infer_params.animation_region in ["all", "pose"]:
                    t_new = x_d_i_info['t'].copy()
                else:
                    t_new = x_s_info['t'].copy()

            if not (self.is_animal and self.cfg.infer_params.flag_relative_motion):
                t_new[..., 2] = 0  # zero tz
                x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new
            if (not self.is_animal and not self.is_source_video and self.cfg.infer_params.flag_relative_motion and
                    self.cfg.infer_params.get("driving_option", "pose-friendly") == "expression-friendly"):
                cache = self.expression_motion_cache.setdefault(j, {})
                if kwargs.get("first_frame", False) or "x_d_0_new" not in cache:
                    cache["x_d_0_new"] = x_d_i_new.copy()
                    cache["motion_multiplier"] = utils.calc_motion_multiplier(x_s, cache["x_d_0_new"])
                x_d_i_new = (x_d_i_new - cache["x_d_0_new"]) * cache["motion_multiplier"] + x_s

            if not self.is_animal:
                # Algorithm 1:
                if not self.cfg.infer_params.flag_stitching and not self.cfg.infer_params.flag_eye_retargeting and not self.cfg.infer_params.flag_lip_retargeting:
                    # without stitching or retargeting
                    if flag_lip_zero and lip_delta_before_animation is not None:
                        x_d_i_new += lip_delta_before_animation.reshape(-1, x_s.shape[1], 3)
                    if self.cfg.infer_params.flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                        x_d_i_new += eye_delta_before_animation
                elif self.cfg.infer_params.flag_stitching and not self.cfg.infer_params.flag_eye_retargeting and not self.cfg.infer_params.flag_lip_retargeting:
                    # with stitching and without retargeting
                    if flag_lip_zero and lip_delta_before_animation is not None:
                        x_d_i_new = self.stitching(x_s, x_d_i_new) + lip_delta_before_animation.reshape(
                            -1, x_s.shape[1], 3)
                    else:
                        x_d_i_new = self.stitching(x_s, x_d_i_new)
                    if self.cfg.infer_params.flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                        x_d_i_new += eye_delta_before_animation
                else:
                    eyes_delta, lip_delta = None, None
                    if self.cfg.infer_params.flag_eye_retargeting:
                        c_d_eyes_i = input_eye_ratio
                        combined_eye_ratio_tensor = self.calc_combined_eye_ratio(c_d_eyes_i,
                                                                                 source_lmk)
                        # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
                        eyes_delta = self.retarget_eye(x_s, combined_eye_ratio_tensor)
                    if self.cfg.infer_params.flag_lip_retargeting:
                        c_d_lip_i = input_lip_ratio
                        combined_lip_ratio_tensor = self.calc_combined_lip_ratio(c_d_lip_i, source_lmk)
                        # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                        lip_delta = self.retarget_lip(x_s, combined_lip_ratio_tensor)

                    if self.cfg.infer_params.flag_relative_motion:  # use x_s
                        x_d_i_new = x_s + \
                                    (eyes_delta.reshape(-1, x_s.shape[1], 3) if eyes_delta is not None else 0) + \
                                    (lip_delta.reshape(-1, x_s.shape[1], 3) if lip_delta is not None else 0)
                    else:  # use x_d,i
                        x_d_i_new = x_d_i_new + \
                                    (eyes_delta.reshape(-1, x_s.shape[1], 3) if eyes_delta is not None else 0) + \
                                    (lip_delta.reshape(-1, x_s.shape[1], 3) if lip_delta is not None else 0)

                    if self.cfg.infer_params.flag_stitching:
                        x_d_i_new = self.stitching(x_s, x_d_i_new)
            else:
                if self.cfg.infer_params.flag_stitching:
                    x_d_i_new = self.stitching(x_s, x_d_i_new)

            x_d_i_new = x_s + (x_d_i_new - x_s) * self.cfg.infer_params.driving_multiplier
            warping_spade = self.model_dict["warping_spade"]
            if getattr(warping_spade, "predict_type", None) == "mlx":
                out_crop = warping_spade.predict(
                    f_s, x_s, x_d_i_new, return_numpy=True, return_uint8=True
                )
            else:
                out_crop = warping_spade.predict(f_s, x_s, x_d_i_new)
            if not realtime and self.cfg.infer_params.flag_pasteback and self.cfg.infer_params.flag_do_crop:
                I_p_pstbk = paste_back_numpy(out_crop, M, I_p_pstbk, mask_ori_float)
        if realtime:
            out_crop_np = _image_to_uint8_numpy(out_crop)
            return out_crop_np, np.asarray(I_p_pstbk, dtype=np.uint8)
        out_crop_np = _image_to_uint8_numpy(out_crop)
        return out_crop_np, np.asarray(I_p_pstbk, dtype=np.uint8)

    def run(self, image, img_src, src_info, **kwargs):
        img_bgr = image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        realtime = kwargs.pop("realtime", False)
        if kwargs.get("first_frame", False):
            self.driving_crop_state = None
        I_p_pstbk = img_src.copy()
        if self.cfg.infer_params.flag_crop_driving_video:
            if self.src_lmk_pre is None:
                src_face = self.model_dict["face_analysis"].predict(img_bgr)
                if len(src_face) == 0:
                    return None, None, None, None
                lmk = src_face[0]
                lmk = self.model_dict["landmark"].predict(img_rgb, lmk)
                self.src_lmk_pre = lmk.copy()
            else:
                lmk = self.model_dict["landmark"].predict(img_rgb, self.src_lmk_pre)
                self.src_lmk_pre = lmk.copy()

            ret_bbox = parse_bbox_from_landmark(
                lmk,
                scale=self.cfg.crop_params.dri_scale,
                vx_ratio=self.cfg.crop_params.dri_vx_ratio,
                vy_ratio=self.cfg.crop_params.dri_vy_ratio,
            )["bbox"]
            global_bbox = [
                ret_bbox[0, 0],
                ret_bbox[0, 1],
                ret_bbox[2, 0],
                ret_bbox[2, 1],
            ]
            global_bbox = self.stabilize_driving_bbox(global_bbox)
            ret_dct = crop_image_by_bbox(
                img_rgb,
                global_bbox,
                lmk=lmk,
                dsize=kwargs.get("dsize", 512),
                flag_rot=False,
                borderValue=(0, 0, 0),
            )
            lmk_crop = ret_dct["lmk_crop"]
            img_crop = ret_dct["img_crop"]
            img_crop = cv2.resize(img_crop, (256, 256))
        else:
            if self.src_lmk_pre is None:
                src_face = self.model_dict["face_analysis"].predict(img_bgr)
                if len(src_face) == 0:
                    return None, None, None, None
                lmk = src_face[0]
                lmk = self.model_dict["landmark"].predict(img_rgb, lmk)
                self.src_lmk_pre = lmk.copy()
            else:
                lmk = self.model_dict["landmark"].predict(img_rgb, self.src_lmk_pre)
                self.src_lmk_pre = lmk.copy()
            lmk_crop = lmk.copy()
            img_crop = cv2.resize(img_rgb, (256, 256))

        input_eye_ratio = calc_eye_close_ratio(lmk_crop[None])
        input_lip_ratio = calc_lip_close_ratio(lmk_crop[None])
        pitch, yaw, roll, t, exp, scale, kp = self.model_dict["motion_extractor"].predict(img_crop)
        x_d_i_info = {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "t": t,
            "exp": exp,
            "scale": scale,
            "kp": kp
        }
        R_d_i = get_rotation_matrix(pitch, yaw, roll)
        x_d_i_info["R"] = R_d_i
        x_d_i_info_copy = copy.deepcopy(x_d_i_info)
        for key in x_d_i_info_copy:
            x_d_i_info_copy[key] = x_d_i_info_copy[key].astype(np.float32)
        dri_motion_info = [x_d_i_info_copy, copy.deepcopy(input_eye_ratio.astype(np.float32)),
                           copy.deepcopy(input_lip_ratio.astype(np.float32))]
        if kwargs.get("first_frame", False) or self.R_d_0 is None:
            self.frame_id = 0
            self.R_d_0 = R_d_i.copy()
            self.x_d_0_info = copy.deepcopy(x_d_i_info)
            warping_spade = self.model_dict.get("warping_spade")
            if hasattr(warping_spade, "reset_temporal_cache"):
                warping_spade.reset_temporal_cache()
            self.expression_motion_cache = {}
            # realtime smooth
            self.R_d_smooth = utils.OneEuroFilter(4, 0.3)
            self.exp_smooth = utils.OneEuroFilter(4, 0.3)
        R_d_0 = self.R_d_0.copy()
        x_d_0_info = copy.deepcopy(self.x_d_0_info)
        out_crop, I_p_pstbk = self._run(src_info, x_d_i_info, x_d_0_info, R_d_i, R_d_0, realtime, input_eye_ratio,
                                        input_lip_ratio,
                                        I_p_pstbk, **kwargs)
        return img_crop, out_crop, I_p_pstbk, dri_motion_info

    def run_with_pkl(self, dri_motion_info, img_src, src_info, **kwargs):
        realtime = kwargs.pop("realtime", False)
        I_p_pstbk = img_src.copy()

        input_eye_ratio = dri_motion_info[1]
        input_lip_ratio = dri_motion_info[2]
        x_d_i_info = dri_motion_info[0]
        R_d_i = x_d_i_info["R"] if "R" in x_d_i_info else x_d_i_info["R_d"]

        if kwargs.get("first_frame", False) or self.R_d_0 is None:
            self.frame_id = 0
            self.R_d_0 = R_d_i.copy()
            self.x_d_0_info = copy.deepcopy(x_d_i_info)
            warping_spade = self.model_dict.get("warping_spade")
            if hasattr(warping_spade, "reset_temporal_cache"):
                warping_spade.reset_temporal_cache()
            self.expression_motion_cache = {}
            # realtime smooth
            self.R_d_smooth = utils.OneEuroFilter(4, 0.3)
            self.exp_smooth = utils.OneEuroFilter(4, 0.3)
        R_d_0 = self.R_d_0.copy()
        x_d_0_info = copy.deepcopy(self.x_d_0_info)
        out_crop, I_p_pstbk = self._run(src_info, x_d_i_info, x_d_0_info, R_d_i, R_d_0, realtime, input_eye_ratio,
                                        input_lip_ratio, I_p_pstbk, **kwargs)
        return out_crop, I_p_pstbk

    def __del__(self):
        if hasattr(self, "model_dict"):
            self.clean_models()
