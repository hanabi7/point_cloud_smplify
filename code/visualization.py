import cv2
from PIL import Image, ImageDraw
import pickle
import cv2
import matplotlib.pyplot as plt
import os.path as osp
import os
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('AGG')
import torch
import trimesh
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as pyplot
import seaborn as sns
import numpy as np

sns.set()
colors_dict = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'black': (105, 105, 105),
    'yellow': (255, 255, 0),
    'purple': (0, 255, 255),
    'grey': (192, 192, 192)
}

color_map = {
    0: 'grey',
    1: 'blue',
    2: 'green',
    3: 'red',
    4: 'purple',
    5: 'black',
    6: 'grey',
    7: 'blue',
    8: 'green',
    9: 'red',
    10: 'purple',
    11: 'black',
    12: 'grey',
    13: 'blue',
    14: 'green',
    15: 'blue',
    16: 'purple',
    17: 'black',
    18: 'grey',
    19: 'blue',
    20: 'green',
    21: 'red',
    22: 'purple',
    23: 'black'
}


def write_mesh(verts, faces, filename, face_colors=None, vertex_colors=None):
    if face_colors is not None:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, face_colors=face_colors)
    elif vertex_colors is not None:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vertex_colors)
    else:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(filename)

def write_joint_cloud(joints, filename):
    verts = []
    for index,joint in enumerate(joints):
        color = colors_dict[color_map[index%24]]
        verts.append((joint[0],joint[1],joint[2],int(color[0]), int(color[1]), int(color[2])))
    verts = np.array(verts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(verts, 'vertex')
    PlyData([el], text=False).write(filename)

def write_point_cloud(vertices, filename, colors=None, labels=None,heatmap = None):
    assert (colors is None or labels is None)
    verts_num = vertices.shape[0]
    if labels is not None:
        labels = labels.astype(int)
        verts = []
        for i in range(verts_num):
            point_color = colors_dict[color_map[labels[i]]]
            verts.append((vertices[i, 0], vertices[i, 1], vertices[i, 2], int(point_color[0]), int(point_color[1]), int(point_color[2])))
        verts = np.array(verts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    elif heatmap is not None:
        verts = []
        cmap = plt.get_cmap('Reds')
        for i in range(verts_num):
            verts_color = cmap(heatmap[i])
            verts.append((vertices[i, 0], vertices[i, 1], vertices[i, 2], int(verts_color[0] * 255), int(verts_color[1] * 255), int(verts_color[2] * 255)))
        verts = np.array(verts,dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    elif colors is not None:
        verts = []
        for i in range(verts_num):
            verts.append((vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]))
        verts = np.array(verts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    else:
        verts = [(vertices[i, 0], vertices[i, 1], vertices[i, 2]) for i in range(verts_num)]
        verts = np.array(verts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(verts, 'vertex')
    PlyData([el], text=False).write(filename)


def write_attention_result(save_dir, point_clouds, attention_value):
    # point_clouds [2500, 3]
    # attention_value [2500, 24]
    key_points_num = attention_value.shape[1]
    points_num = attention_value.shape[0]
    if not osp.exists(save_dir):
        os.mkdir(save_dir)
    for i in range(key_points_num):
        part_attention_value = attention_value[:, i]
        # [2500]
        part_file_name = save_dir + str(i) + '_body_attention.ply'
        part_color_map = [color_map(part_attention_value[i]) for i in range(points_num)]
        write_point_cloud(point_clouds, part_file_name, colors=part_color_map)

