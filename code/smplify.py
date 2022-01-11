from typing import List, Tuple, Union
import torch
from mmcv.runner import build_optimizer
import os
import sys
base_path = '/data/liuguanze/baseline/prox_process'
sys.path.append(base_path)
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
from mmhuman3d.models.builder import build_body_model, build_loss


class OptimizableParameters():
    def __init__(self):
        self.opt_params = []
    
    def set_param(self, fit_param, param):
        if fit_param:
            param.requires_grad = True
            self.opt_params.append(param)
        else:
            param.requires_grad = False
    
    def parameters(self):
        return self.opt_params


class SMPLify(object):
    def __init__(
        self,
        body_model: dict,
        num_epoches: int=20, 
        stages: dict=None, 
        optimizer: dict=None,
        chamfer_loss: dict=None,
        pose_prior_loss: dict=None,
        shape_prior_loss: dict=None,
        joint_prior_loss: dict=None, 
        smooth_loss: dict=None,
        use_one_betas_per_videos: bool=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose: bool=False):
        self.use_one_betas_per_videos = use_one_betas_per_videos
        self.num_epochs = num_epoches
        self.optimizer = optimizer
        self.stage_config = stages
        self.device = device
        self.verbose = verbose
        self.chamfer_loss = dist_chamfer_3D.chamfer_3DDist()
        self.shape_prior_loss = build_loss(shape_prior_loss).to(device)
        self.pose_prior_loss = build_loss(pose_prior_loss).to(device)
        self.joint_prior_loss = build_loss(joint_prior_loss).to(device)
        self.smooth_loss = build_loss(smooth_loss).to(device)
        if self.joint_prior_loss is not None:
            self.joint_prior_loss = self.joint_prior_loss.to(device)
        if self.shape_prior_loss is not None:
            self.shape_prior_loss = self.shape_prior_loss.to(device)
        if self.smooth_loss is not None:
            self.smooth_loss = self.smooth_loss.to(device)
        self.body_model = build_body_model(body_model).to(device)

    def __call__(self,
                 human_point_clouds: torch.Tensor = None, 
                 init_global_orient: torch.Tensor = None,
                 init_transl: torch.Tensor = None,
                 init_body_pose: torch.Tensor = None,
                 init_betas: torch.Tensor = None,
                 chamfer_loss_weight: float = None,
                 joint_prior_weight: float = None,
                 shape_prior_weight: float = None,
                 smooth_loss_weight: float = None,
                 pose_prior_weight: float = None,
                 return_verts: bool = False,
                 return_joints: bool = False,
                 return_full_pose: bool = False,
                 return_losses: bool = False) -> dict:
        human_point_clouds = human_point_clouds.detach().clone()
        global_orient = init_global_orient.detach().clone() \
            if init_global_orient is not None \
            else self.body_model.global_orient.detach().clone()
        transl = init_transl.detach().clone() \
            if init_transl is not None \
            else self.body_model.transl.detach().clone()
        body_pose = init_body_pose.detach().clone() \
            if init_body_pose is not None \
            else self.body_model.body_pose.detach().clone()
        if init_betas is not None:
            betas = init_betas.detach().clone()
        elif self.use_one_betas_per_video:
            betas = torch.zeros(1, self.body_model.betas.shape[-1]).to(
                self.device)
        else:
            betas = self.body_model.betas.detach().clone()
        for i in range(self.num_epochs):
            for stage_idx, stage_config in enumerate(self.stage_config):
                self._optimize_stage(
                    global_orient=global_orient,
                    transl=transl,
                    body_pose=body_pose,
                    betas=betas,
                    human_point_clouds=human_point_clouds,
                    chamfer_loss_weight=chamfer_loss_weight,
                    shape_prior_weight=shape_prior_weight,
                    joint_prior_weight=joint_prior_weight,
                    smooth_loss_weight=smooth_loss_weight,
                    pose_prior_weight=pose_prior_weight,
                    **stage_config
                )

        # collate results
        ret = {
            'global_orient': global_orient,
            'transl': transl,
            'body_pose': body_pose,
            'betas': betas
        }

        eval_ret = self.evaluate(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
            human_point_clouds=human_point_clouds,
            chamfer_loss_weight=chamfer_loss_weight,
            shape_prior_weight=shape_prior_weight,
            joint_prior_weight=joint_prior_weight,
            smooth_loss_weight=smooth_loss_weight,
            pose_prior_weight=pose_prior_weight,
            return_verts=True,
            return_full_pose=True,
            return_joints=True,
            reduction_override='none'  # sample-wise loss
        )
        # import ipdb; ipdb.set_trace()
        ret['vertices'] = eval_ret['vertices']
        ret['joints'] = eval_ret['joints']
        ret['full_pose'] = eval_ret['full_pose']
        if return_losses:
            for k in eval_ret.keys():
                if 'loss' in k:
                    ret[k] = eval_ret[k]
        for k, v in ret.items():
            if isinstance(v, torch.Tensor):
                ret[k] = v.detach().clone()
        return ret

    def _expand_betas(self, batch_size, betas):
        """A helper function to expand the betas's first dim to match batch
        size such that the same beta parameters can be used for all frames in a
        video sequence.

        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension

        Args:
            batch_size: batch size
            betas: shape (B, D)

        Returns:
            betas_video: expanded betas
        """
        # no expansion needed
        if batch_size == betas.shape[0]:
            return betas

        # first dim is 1
        else:
            feat_dim = betas.shape[-1]
            betas_video = betas.view(1, feat_dim).expand(batch_size, feat_dim)

        return betas_video

    def _optimize_stage(self,
                    betas: torch.Tensor,
                    body_pose: torch.Tensor,
                    global_orient: torch.Tensor,
                    transl: torch.Tensor,
                    human_point_clouds: torch.Tensor, 
                    fit_global_orient: bool = True,
                    fit_transl: bool = True,
                    fit_body_pose: bool = True,
                    fit_betas: bool = True,
                    chamfer_loss_weight: float=None,
                    shape_prior_weight: float = None,
                    joint_prior_weight: float = None,
                    smooth_loss_weight: float = None,
                    pose_prior_weight: float = None,
                    num_iter: int = 1) -> None:

        parameters = OptimizableParameters()
        parameters.set_param(fit_global_orient, global_orient)
        parameters.set_param(fit_transl, transl)
        parameters.set_param(fit_body_pose, body_pose)
        parameters.set_param(fit_betas, betas)

        optimizer = build_optimizer(parameters, self.optimizer)

        for iter_idx in range(num_iter):
            def closure():
                optimizer.zero_grad()
                betas_video = self._expand_betas(body_pose.shape[0], betas)

                loss_dict = self.evaluate(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    betas=betas_video,
                    transl=transl,
                    human_point_clouds=human_point_clouds,
                    chamfer_loss_weight=chamfer_loss_weight,
                    joint_prior_weight=joint_prior_weight,
                    shape_prior_weight=shape_prior_weight,
                    smooth_loss_weight=smooth_loss_weight,
                    pose_prior_weight=pose_prior_weight)
                loss = loss_dict['total_loss']
                loss.backward()
                return loss

            optimizer.step(closure)

    def evaluate(
        self,
        human_point_clouds: torch.Tensor=None,
        betas: torch.Tensor = None,
        body_pose: torch.Tensor = None,
        global_orient: torch.Tensor = None,
        transl: torch.Tensor = None,
        chamfer_loss_weight: float=None,
        shape_prior_weight: float = None,
        joint_prior_weight: float = None,
        smooth_loss_weight: float = None,
        pose_prior_weight: float = None,
        joint_weights: dict = {},
        return_verts: bool = False,
        return_full_pose: bool = False,
        return_joints: bool = False,
        reduction_override: str = None,
    ) -> dict:
        ret = {}
        body_model_output = self.body_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
            return_verts=True,
            return_transl=True,
            return_full_pose=True,
            return_joints=True)

        model_vertices = body_model_output['vertices']
        loss_dict = self._compute_loss(
            model_vertices=model_vertices,
            human_point_clouds=human_point_clouds,
            chamfer_distance_weight=chamfer_loss_weight,
            joint_prior_weight=joint_prior_weight,
            shape_prior_weight=shape_prior_weight,
            smooth_loss_weight=smooth_loss_weight,
            pose_prior_weight=pose_prior_weight,
            reduction_override=reduction_override,
            body_pose=body_pose,
            betas=betas)
        ret.update(loss_dict)
        # import ipdb; ipdb.set_trace()
        if return_joints:
            ret['joints'] = body_model_output['joints']
        if return_verts:
            ret['vertices'] = body_model_output['vertices']
        if return_full_pose:
            ret['full_pose'] = body_model_output['full_pose']
        return ret

    def _compute_loss(self,
                      model_vertices: torch.Tensor,
                      human_point_clouds: torch.Tensor,
                      chamfer_distance_weight: float = None,
                      shape_prior_weight: float = None,
                      joint_prior_weight: float = None,
                      smooth_loss_weight: float = None,
                      pose_prior_weight: float = None,
                      reduction_override: str = None,
                      body_pose: torch.Tensor = None,
                      betas: torch.Tensor = None):
        losses = {}

        # chamfer Distance loss
        if self.chamfer_loss is not None:
            human_point_clouds = human_point_clouds.unsqueeze(0)
            dist1, dist2, idx1, idx2 = self.chamfer_loss(human_point_clouds, model_vertices)
            chamfer_loss = torch.mean(dist1)
            chamfer_loss *= chamfer_distance_weight
            losses['chamfer_loss'] = chamfer_loss

        # regularizer to prevent betas from taking large values
        if self.shape_prior_loss is not None:
            shape_prior_loss = self.shape_prior_loss(
                betas=betas,
                loss_weight_override=shape_prior_weight,
                reduction_override=reduction_override)
            losses['shape_prior_loss'] = shape_prior_loss

        # joint prior loss
        if self.joint_prior_loss is not None:
            joint_prior_loss = self.joint_prior_loss(
                body_pose=body_pose,
                loss_weight_override=joint_prior_weight,
                reduction_override=reduction_override)
            losses['joint_prior_loss'] = joint_prior_loss

        # smooth body loss
        if self.smooth_loss is not None:
            smooth_loss = self.smooth_loss(
                body_pose=body_pose,
                loss_weight_override=smooth_loss_weight,
                reduction_override=reduction_override)
            losses['smooth_loss'] = smooth_loss

        # pose prior loss
        if self.pose_prior_loss is not None:
            pose_prior_loss = self.pose_prior_loss(
                body_pose=body_pose,
                loss_weight_override=pose_prior_weight,
                reduction_override=reduction_override)
            losses['pose_prior_loss'] = pose_prior_loss
        # import ipdb; ipdb.set_trace()
        if self.verbose:
            msg = ''
            for loss_name, loss in losses.items():
                msg += f'{loss_name}={loss.mean().item():.6f}'
            print(msg)

        total_loss = 0
        for loss_name, loss in losses.items():
            if loss.ndim == 3:
                total_loss = total_loss + loss.sum(dim=(2, 1))
            elif loss.ndim == 2:
                total_loss = total_loss + loss.sum(dim=-1)
            else:
                total_loss = total_loss + loss
        losses['total_loss'] = total_loss

        return losses

