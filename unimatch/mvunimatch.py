import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .transformer import FeatureTransformer
from .mvmatching import (global_correlation_softmax_stereo, local_correlation_softmax_stereo, local_correlation_with_flow)
from .attention import SelfAttnPropagation
from .geometry import flow_warp
from .reg_refine import BasicUpdateBlock
from .utils import feature_add_position, upsample_flow_with_mask


class MVUniMatch(nn.Module):
    def __init__(self,
                 num_scales=1,
                 feature_channels=128,
                 upsample_factor=8,
                 num_head=1,
                 ffn_dim_expansion=4,
                 num_transformer_layers=6,
                 reg_refine=False,  # optional local regression refinement
                 task='flow',
                 ):
        super(MVUniMatch, self).__init__()

        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.upsample_factor = upsample_factor
        self.reg_refine = reg_refine

        # CNN
        self.backbone = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)

        # Transformer
        self.transformer = FeatureTransformer(num_layers=num_transformer_layers,
                                              d_model=feature_channels,
                                              nhead=num_head,
                                              ffn_dim_expansion=ffn_dim_expansion,
                                              )

        # propagation with self-attn
        self.feature_flow_attn = SelfAttnPropagation(in_channels=feature_channels)

        if not self.reg_refine or task == 'depth':
            # convex upsampling simiar to RAFT
            # concat feature0 and low res flow as input
            self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))
            # thus far, all the learnable parameters are task-agnostic

        if reg_refine:
            # optional task-specific local regression refinement
            self.refine_proj = nn.Conv2d(128, 256, 1)
            self.refine = BasicUpdateBlock(corr_channels=(2 * 4 + 1) ** 2,
                                           downsample_factor=upsample_factor,
                                           flow_dim=2 if task == 'flow' else 1,
                                           bilinear_up=task == 'depth',
                                           )

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        features = self.backbone(concat)  # list of [2B, C, H, W], resolution from high to low

        # reverse: resolution from low to high
        features = features[::-1]

        feature0, feature1 = [], []

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)  # tuple
            feature0.append(chunks[0])
            feature1.append(chunks[1])

        return feature0, feature1

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8,
                      is_depth=False):
        if bilinear:
            multiplier = 1 if is_depth else upsample_factor
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * multiplier
        else:
            concat = torch.cat((flow, feature), dim=1)
            mask = self.upsampler(concat)
            up_flow = upsample_flow_with_mask(flow, mask, upsample_factor=self.upsample_factor,
                                              is_depth=is_depth)

        return up_flow

    def forward(
        self, 
        images,
        calibs,
        attn_type=None,
        attn_splits_list=None,
        corr_radius_list=None,
        prop_radius_list=None,
        num_reg_refine=1,
        pred_bidir_flow=False,
        task='flow',
        intrinsics=None,
        pose=None,  # relative pose transform
        min_depth=1. / 0.5,  # inverse depth range
        max_depth=1. / 10,
        num_depth_candidates=64,
        depth_from_argmax=False,
        pred_bidir_depth=False,
        **kwargs,
    ):

        results_dict = {}
        flow_preds = []

        # list of features, resolution low to high
        features_left = []
        features_right = []
        for image_pair in images:
            feature0_list, feature1_list = self.extract_feature(image_pair[0], image_pair[1])  # list of features
            features_left.append(feature0_list)
            features_right.append(feature1_list)

        flow = None
        # assert len(attn_splits_list) == len(corr_radius_list) == len(prop_radius_list) == self.num_scales

        for scale_idx in range(self.num_scales):
            features_left_sidx = [fx[scale_idx] for fx in features_left]
            features_right_sidx = [fx[scale_idx] for fx in features_right]

            features_left_ori, features_right_ori = features_left_sidx, features_right_sidx

            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))

            if scale_idx > 0:
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2

            if flow is not None:
                # raise NotImplementedError
                flow = flow.detach()
                # construct flow vector for disparity
                # flow here is actually disparity
                zeros = torch.zeros_like(flow)  # [B, 1, H, W]
                # NOTE: reverse disp, disparity is positive
                displace = torch.cat((-flow, zeros), dim=1)  # [B, 2, H, W]
                features_right_sidx[0] = flow_warp(features_right_sidx[0], displace)  # [B, C, H, W]

            attn_splits = attn_splits_list[scale_idx]
            corr_radius = corr_radius_list[scale_idx]
            prop_radius = prop_radius_list[scale_idx]

            for bid in range(len(features_left)):
                # add position to features
                features_left_sidx[bid], features_right_sidx[bid] = feature_add_position(
                    features_left_sidx[bid], features_right_sidx[bid],
                    attn_splits, 
                    self.feature_channels,
                )

                # Transformer
                features_left_sidx[bid], features_right_sidx[bid] = self.transformer(
                    features_left_sidx[bid], features_right_sidx[bid],
                    attn_type=attn_type,
                    attn_num_splits=attn_splits,
                )

            # correlation and softmax
            if corr_radius == -1:  # global matching
                flow_pred = global_correlation_softmax_stereo(features_left_sidx, features_right_sidx, calibs)[0]
            else:  # local matching
                # raise NotImplementedError
                flow_pred = local_correlation_softmax_stereo(features_left_sidx[0], features_right_sidx[0], corr_radius)[0]

            # flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred

            flow = flow.clamp(min=0)  # positive disparity

            # upsample to the original resolution for supervison at training time only
            if self.training:
                raise NotImplementedError
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor,
                                                   is_depth=task == 'depth')
                flow_preds.append(flow_bilinear)

            flow = self.feature_flow_attn(features_left_sidx[0], flow.detach(),
                                          local_window_attn=prop_radius > 0,
                                          local_window_radius=prop_radius,
                                          )

            # bilinear exclude the last one
            if self.training and scale_idx < self.num_scales - 1:
                raise NotImplementedError
                flow_up = self.upsample_flow(flow, features_left_sidx[0], bilinear=True,
                                             upsample_factor=upsample_factor,
                                             is_depth=task == 'depth')
                flow_preds.append(flow_up)

            if scale_idx == self.num_scales - 1:
                if not self.reg_refine:
                    # upsample to the original image resolution
                    flow_pad = torch.cat((-flow, torch.zeros_like(flow)), dim=1)  # [B, 2, H, W]
                    flow_up_pad = self.upsample_flow(flow_pad, features_left_sidx[0])
                    flow_up = -flow_up_pad[:, :1]  # [B, 1, H, W]

                    flow_preds.append(flow_up)
                else:
                    # task-specific local regression refinement
                    # supervise current flow
                    if self.training:
                        flow_up = self.upsample_flow(flow, features_left_sidx[0], bilinear=True,
                                                     upsample_factor=upsample_factor,
                                                     is_depth=task == 'depth')
                        flow_preds.append(flow_up)

                    assert num_reg_refine > 0
                    for refine_iter_idx in range(num_reg_refine):
                        flow = flow.detach()

                        zeros = torch.zeros_like(flow)  # [B, 1, H, W]
                        # NOTE: reverse disp, disparity is positive
                        displace = torch.cat((-flow, zeros), dim=1)  # [B, 2, H, W]
                        correlation = local_correlation_with_flow(
                            features_left_ori[0],
                            features_right_ori[0],
                            flow=displace,
                            local_radius=4,
                        )  # [B, (2R+1)^2, H, W]
                        
                        proj = self.refine_proj(features_left_sidx[0])

                        net, inp = torch.chunk(proj, chunks=2, dim=1)

                        net = torch.tanh(net)
                        inp = torch.relu(inp)

                        net, up_mask, residual_flow = self.refine(net, inp, correlation, flow.clone())

                        flow = flow + residual_flow

                        flow = flow.clamp(min=0)  # positive

                        if self.training or refine_iter_idx == num_reg_refine - 1:
                            flow_up = upsample_flow_with_mask(flow, up_mask, upsample_factor=self.upsample_factor,
                                                                  is_depth=task == 'depth')

                            flow_preds.append(flow_up)

        for i in range(len(flow_preds)):
            flow_preds[i] = flow_preds[i].squeeze(1)  # [B, H, W]

        results_dict.update({'flow_preds': flow_preds})

        return results_dict
