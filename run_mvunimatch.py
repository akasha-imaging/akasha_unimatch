import argparse
import glob
import os
import pickle

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils import misc
from unimatch.mvunimatch import MVUniMatch
from dataloader.stereo import transforms
from utils.visualization import vis_disparity

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

@torch.no_grad()
def inference_mvstereo(
    model,
    inference_dir=None,
    padding_factor=16,
    inference_size=None,
    attn_type=None,
    attn_splits_list=None,
    corr_radius_list=None,
    prop_radius_list=None,
    num_reg_refine=1,
    pred_bidir_disp=False,
    pred_right_disp=False,
    save_pfm_disp=False,
):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    val_transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    val_transform = transforms.Compose(val_transform_list)
    
    assert inference_dir 
    baselines = glob.glob(os.path.join(inference_dir, '*'))

    fixed_inference_size = inference_size
    images = []
    calibs = []
    with open(os.path.join(inference_dir, 'postcalib_clearsights.pkl'), 'rb') as file:
        calib_input = pickle.load(file)

    for baseline in baselines:
        if not os.path.isdir(baseline):
            continue

        image_left = np.array(Image.open(os.path.join(baseline, 'image_left.png')).convert('RGB')).astype(np.float32)
        image_right = np.array(Image.open(os.path.join(baseline, 'image_right.png')).convert('RGB')).astype(np.float32)
        # image_left = cv2.imread(os.path.join(baseline, 'image_left.png'))
        # image_right = cv2.imread(os.path.join(baseline, 'image_left.png'))

        sample = {'left': image_left, 'right': image_right}
        sample = val_transform(sample)

        left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]

        nearest_size = [
            int(np.ceil(left.size(-2) / padding_factor)) * padding_factor,
            int(np.ceil(left.size(-1) / padding_factor)) * padding_factor,
        ]

        # resize to nearest size or specified size
        inference_size = nearest_size if fixed_inference_size is None else fixed_inference_size

        ori_size = left.shape[-2:]
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            left = F.interpolate(left, size=inference_size,
                                 mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size,
                                  mode='bilinear',
                                  align_corners=True)
        
        images.append([left, right])
        device = images[0][0]

        left_camera_id = os.path.basename(baseline).split('-')[0].split('_')[0]
        left_camera_type = os.path.basename(baseline).split('-')[0].split('_')[-1]
        right_camera_id = os.path.basename(baseline).split('-')[1].split('_')[0]
        right_camera_type = os.path.basename(baseline).split('-')[1].split('_')[-1]

        Ks = [
            torch.from_numpy(calib_input['clearsights'][left_camera_id]['cameras'][f'left_{left_camera_type}']['K']).to(device),
            torch.from_numpy(calib_input['clearsights'][right_camera_id]['cameras'][f'right_{right_camera_type}']['K']).to(device),
        ]
        Poses = [
            torch.from_numpy(calib_input['clearsights'][left_camera_id]['cameras'][f'left_{left_camera_type}']['pose']).to(device),
            torch.from_numpy(calib_input['clearsights'][right_camera_id]['cameras'][f'right_{right_camera_type}']['pose']).to(device),
        ]
        calibs.append([Ks, Poses])

    with torch.no_grad():
        pred_disp = model(
            images,
            calibs,
            attn_type=attn_type,
            attn_splits_list=attn_splits_list,
            corr_radius_list=corr_radius_list,
            prop_radius_list=prop_radius_list,
            num_reg_refine=num_reg_refine,
            task='stereo',
        )['flow_preds'][-1]

    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        # resize back
        pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size,
                                    mode='bilinear',
                                    align_corners=True).squeeze(1)  # [1, H, W]
        pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])
    
    disp = pred_disp[0].cpu().numpy()

    return disp

def main(args):
    misc.check_path(args.output_path)
    torch.backends.cudnn.benchmark = True

    if args.launcher == 'none':
        args.distributed = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # init model
    model = MVUniMatch(
        feature_channels=args.feature_channels,
        num_scales=args.num_scales,
        upsample_factor=args.upsample_factor,
        num_head=args.num_head,
        ffn_dim_expansion=args.ffn_dim_expansion,
        num_transformer_layers=args.num_transformer_layers,
        reg_refine=args.reg_refine,
        task=args.task,
    ).to(device)

    # load weights
    print("=> Load checkpoint: %s" % args.resume)
    loc = 'cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(args.resume, map_location=loc)
    model.load_state_dict(checkpoint['model'], strict=args.strict_resume)

    disp = inference_mvstereo(
        model,
        inference_dir=args.inference_dir,
        padding_factor=args.padding_factor,
        inference_size=args.inference_size,
        attn_type=args.attn_type,
        attn_splits_list=args.attn_splits_list,
        corr_radius_list=args.corr_radius_list,
        prop_radius_list=args.prop_radius_list,
        num_reg_refine=args.num_reg_refine,
        pred_bidir_disp=args.pred_bidir_disp,
        pred_right_disp=args.pred_right_disp,
        save_pfm_disp=args.save_pfm_disp,
    )
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    save_name = os.path.join(args.output_path, 'disp.png')
    disp = vis_disparity(disp)
    cv2.imwrite(save_name, disp)
    print('Done!')

def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoint_dir', default='tmp', type=str,
                        help='where to save the training log and models')
    parser.add_argument('--stage', default='sceneflow', type=str,
                        help='training stage on different datasets')
    parser.add_argument('--val_dataset', default=['kitti15'], type=str, nargs='+')
    parser.add_argument('--max_disp', default=400, type=int,
                        help='exclude very large disparity in the loss function')
    parser.add_argument('--img_height', default=288, type=int)
    parser.add_argument('--img_width', default=512, type=int)
    parser.add_argument('--padding_factor', default=16, type=int)

    # training
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--seed', default=326, type=int)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrained model or resume from unexpectedly terminated training')
    parser.add_argument('--strict_resume', action='store_true',
                        help='strict resume while loading pretrained weights')
    parser.add_argument('--no_resume_optimizer', action='store_true')
    parser.add_argument('--resume_exclude_upsampler', action='store_true')

    # model: learnable parameters
    parser.add_argument('--task', default='stereo', choices=['flow', 'stereo', 'depth'], type=str)
    parser.add_argument('--num_scales', default=1, type=int,
                        help='feature scales: 1/8 or 1/8 + 1/4')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--reg_refine', action='store_true',
                        help='optional task-specific local regression refinement')

    # model: parameter-free
    parser.add_argument('--attn_type', default='self_swin2d_cross_1d', type=str,
                        help='attention function')
    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                        help='number of splits in attention')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                        help='correlation radius for matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='self-attention radius for propagation, -1 indicates global attention')
    parser.add_argument('--num_reg_refine', default=1, type=int,
                        help='number of additional local regression refinement')

    # evaluation
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--inference_size', default=None, type=int, nargs='+')
    parser.add_argument('--count_time', action='store_true')
    parser.add_argument('--save_vis_disp', action='store_true')
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--middlebury_resolution', default='F', choices=['Q', 'H', 'F'])

    # submission
    parser.add_argument('--submission', action='store_true')
    parser.add_argument('--eth_submission_mode', default='train', type=str, choices=['train', 'test'])
    parser.add_argument('--middlebury_submission_mode', default='training', type=str, choices=['training', 'test'])
    parser.add_argument('--output_path', default='output', type=str)

    # log
    parser.add_argument('--summary_freq', default=100, type=int, help='Summary frequency to tensorboard (iterations)')
    parser.add_argument('--save_ckpt_freq', default=1000, type=int, help='Save checkpoint frequency (steps)')
    parser.add_argument('--val_freq', default=1000, type=int, help='validation frequency in terms of training steps')
    parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)
    parser.add_argument('--num_steps', default=100000, type=int)

    # distributed training
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--launcher', default='none', type=str)
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    # inference
    parser.add_argument('--inference_dir', default=None, type=str)
    parser.add_argument('--inference_dir_left', default=None, type=str)
    parser.add_argument('--inference_dir_right', default=None, type=str)
    parser.add_argument('--pred_bidir_disp', action='store_true',
                        help='predict both left and right disparities')
    parser.add_argument('--pred_right_disp', action='store_true',
                        help='predict right disparity')
    parser.add_argument('--save_pfm_disp', action='store_true',
                        help='save predicted disparity as .pfm format')

    parser.add_argument('--debug', action='store_true')

    return parser

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    main(args)
