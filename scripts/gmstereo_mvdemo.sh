#!/usr/bin/env bash


# gmstereo-scale2-regrefine3 model
CUDA_VISIBLE_DEVICES=0 python run_mvunimatch.py \
--inference_dir demo/stereo-intrinsic \
--inference_size 1024 1920 \
--output_path output/gmstereo-scale2-regrefine3-intrinsic \
--resume pretrained/gmstereo-scale2-regrefine3-resumeflowthings-mixdata-train320x640-ft640x960-e4e291fd.pth \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_type self_swin2d_cross_swin1d \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 3

# optionally predict both left and right disparities
#--pred_bidir_disp