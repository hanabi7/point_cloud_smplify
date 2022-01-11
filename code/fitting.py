import torch
import os
import numpy as np
import os.path as osp
import pickle
import json
import argparse
from smplify import SMPLify 
from visualization import write_point_cloud
from mmhuman3d.utils.geometry import batch_rodrigues, rotation_matrix_to_angle_axis


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/data/liuguanze/datasets/prox_processed/')
    parser.add_argument('--chamfer_loss_weight', type=float, default=100)
    parser.add_argument('--joint_prior_weight', type=float, default=0)
    parser.add_argument('--shape_prior_weight', type=float, default=1)
    parser.add_argument('--smooth_loss_weight', type=float, default=0)
    parser.add_argument('--pose_prior_weight', type=float, default=0)
    parser.add_argument('--num_epoches', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--use_one_betas_per_videos', action='store_true', help='whether to use one betas per video sequence')
    return parser.parse_args()


def tensor_based_global_orient_process(origin_global_orient, R):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    origin_global_orient_matrix = batch_rodrigues(origin_global_orient)
    origin_global_orient_matrix = origin_global_orient_matrix.reshape(3, 3)
    refine_global_orient_matrix = R.mm(origin_global_orient_matrix)
    refine_global_orient_matrix_reshape = refine_global_orient_matrix.reshape(1, 3, 3)
    refine_global_orient_matrix_hom = torch.cat([refine_global_orient_matrix_reshape, torch.tensor([0, 0, 1], dtype=torch.float32, device=device).view(1, 3, 1)], dim=-1)
    refine_global_orient = rotation_matrix_to_angle_axis(refine_global_orient_matrix_hom)
    return refine_global_orient


def process_data(args):
    dataset_path = args.dataset_path
    body_model_dict = dict(
        type='SMPL',
        gender='neutral',
        num_betas=10,
        keypoint_src='smpl_24',
        keypoint_dst='smpl_24',
        model_path='data/body_models/smpl',
        batch_size=1
    )
    stages = [
        # stage 1 
        dict(
            num_iter=5,
            fit_global_orient=False,
            fit_transl=True,
            fit_body_pose=False,
            fit_betas=False
        ),
        dict(
            num_iter=15,
            fit_global_orient=True,
            fit_transl=True,
            fit_body_pose=True,
            fit_betas=True
        )
    ]
    optimizer_dict = dict(
        type='LBFGS', max_iter=20, lr=1e-2, line_search_fn='strong_wolfe'
    )
    shape_prior_loss = dict(
        type='ShapePriorLoss', loss_weight=1, reduction='sum'
    )
    joint_prior_loss = dict(
        type='JointPriorLoss',
        loss_weight=20,
        reduction='sum',
        smooth_spine=True,
        smooth_spine_loss_weight=20,
        use_full_body=True
    )
    smooth_loss = dict(
        type='SmoothJointLoss', loss_weight=0, reduction='sum'
    )
    chamfer_loss = dict(
        type='chamfer_loss', loss_weight=0, reduction='sum'
    )
    pose_prior_loss = dict(
    type='MaxMixturePrior',
    prior_folder='data',
    num_gaussians=8,
    loss_weight=4.78**2,
    reduction='sum')
    scenes_list = os.listdir(dataset_path)
    smplify_model = SMPLify(
        body_model=body_model_dict,
        num_epoches=args.num_epoches,
        optimizer=optimizer_dict,
        chamfer_loss=chamfer_loss,
        stages=stages,
        pose_prior_loss=pose_prior_loss, 
        shape_prior_loss=shape_prior_loss, 
        joint_prior_loss=joint_prior_loss,
        smooth_loss=smooth_loss,
        use_one_betas_per_videos=args.use_one_betas_per_videos
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cam2world_dir = '/data/liuguanze/datasets/cam2world/'
    for scene_name in scenes_list:
        scene_dir = osp.join(dataset_path, scene_name)
        scene = scene_name.split('_')[0]
        scene_cam2world_dir = osp.join(cam2world_dir, scene + '.json')
        pkl_file_list = os.listdir(scene_dir)
        for pkl_file in pkl_file_list:
            pkl_dir = osp.join(scene_dir, pkl_file)
            with open(pkl_dir, 'rb') as f:
                pkl_data = pickle.load(f, encoding='bytes')
            human_point_clouds = pkl_data['human_points']
            human_point_clouds = torch.from_numpy(human_point_clouds).to(device)
            origin_body_pose = pkl_data['body_pose']
            origin_body_pose = torch.from_numpy(origin_body_pose).to(device)
            origin_body_shape = pkl_data['gt_shape']
            origin_body_shape = torch.from_numpy(origin_body_shape).to(device)
            origin_transl = pkl_data['transl']
            origin_transl = torch.from_numpy(origin_transl).to(device)
            origin_global_orient = pkl_data['global_orient']
            origin_global_orient = torch.from_numpy(origin_global_orient)
            with open(scene_cam2world_dir, 'rb') as f:
                cam2world = json.load(f)
            cam2world = np.array(cam2world)
            R = torch.tensor(cam2world[:3, :3].reshape(3, 3), dtype=torch.float32).to(device)
            origin_global_orient = origin_global_orient.to(device)
            origin_world_cordinates_global_orient = tensor_based_global_orient_process(origin_global_orient, R)
            return_dict = smplify_model(
                human_point_clouds=human_point_clouds,
                init_global_orient=origin_world_cordinates_global_orient,
                init_transl=None,
                init_body_pose=None,
                init_betas=origin_body_shape,
                return_verts=True,
                return_joints=True,
                return_full_pose=True,
                return_losses=False,
                chamfer_loss_weight=args.chamfer_loss_weight,
                joint_prior_weight=args.joint_prior_weight,
                shape_prior_weight=args.shape_prior_weight,
                smooth_loss_weight=args.smooth_loss_weight,
                pose_prior_weight=args.pose_prior_weight
            )
            pkl_data['refined_pose'] = return_dict['body_pose']
            pkl_data['refined_global_orient'] = return_dict['global_orient']
            pkl_data['refined_vertices'] = return_dict['vertices']
            origin_vertices_name = '/data/liuguanze/baseline/visualization/origin_vertices.ply'
            vertices_file_name = '/data/liuguanze/baseline/visualization/refined_vertices.ply'
            human_point_file_name = '/data/liuguanze/baseline/visualization/human_points.ply'
            write_point_cloud(return_dict['vertices'].reshape(6890, 3), vertices_file_name)
            write_point_cloud(pkl_data['human_points'], human_point_file_name)
            write_point_cloud(pkl_data['vertices'], origin_vertices_name)
            print('Visualization:')


if __name__ == '__main__':
    args = get_argparser()
    process_data(args)
