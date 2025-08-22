import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

__all__ = ['visualize']

# Constants
NOT_OBSERVED = -1
FREE = 0
OCCUPIED = 1
FREE_LABEL = 17
BINARY_OBSERVED = 1
BINARY_NOT_OBSERVED = 0

VOXEL_SIZE = [0.4, 0.4, 0.4]
POINT_CLOUD_RANGE = [-40, -40, -1, 40, 40, 5.4]
SPTIAL_SHAPE = [200, 200, 16]
TGT_VOXEL_SIZE = [0.4, 0.4, 0.4]
TGT_POINT_CLOUD_RANGE = [-40, -40, -1, 40, 40, 5.4]

# Color map for occupancy classes
colormap_to_colors = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [112, 128, 144, 255],  # 1 barrier
        [220, 20, 60, 255],    # 2 bicycle
        [255, 127, 80, 255],   # 3 bus
        [255, 158, 0, 255],  # 4 car
        [233, 150, 70, 255],   # 5 cons. Veh
        [255, 61, 99, 255],  # 6 motorcycle
        [0, 0, 230, 255], # 7 pedestrian
        [47, 79, 79, 255],  # 8 traffic cone
        [255, 140, 0, 255],# 9 trailer
        [255, 99, 71, 255],# 10 truck
        [0, 207, 191, 255],    # 11 drive sur
        [175, 0, 75, 255],  # 12 other lat
        [75, 0, 75, 255],  # 13 sidewalk
        [112, 180, 60, 255],    # 14 terrain
        [222, 184, 135, 255],    # 15 manmade
        [0, 175, 0, 255],   # 16 vegetation
], dtype=np.float32)

# Define color_map for occ2img function (using the same colors as colormap_to_colors)
color_map = colormap_to_colors.astype(np.uint8)

# Define occ_class_names for occ2img function
occ_class_names = [
    'undefined', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

# Define inst_class_ids for occ2img function
inst_class_ids = [2, 3, 4, 5, 6, 7, 9, 10]

# Generate pano_color_map for occ2img function
def generate_rgb_color(number):
    red = (number % 256)
    green = ((number // 256) % 256)
    blue = ((number // 65536) % 256)
    return [red, green, blue]

pano_color_map = np.array([generate_rgb_color(number) for number in np.random.randint(0, 65536*256, 256)])

def voxel2points(voxel, occ_show, voxelSize):
    """
    Args:
        voxel: (Dx, Dy, Dz)
        occ_show: (Dx, Dy, Dz)
        voxelSize: (dx, dy, dz)

    Returns:
        points: (N, 3) 3: (x, y, z)
        voxel: (N, ) cls_id
        occIdx: (x_idx, y_idx, z_idx)
    """
    occIdx = torch.where(occ_show)
    points = torch.cat((occIdx[0][:, None] * voxelSize[0] + POINT_CLOUD_RANGE[0], \
                        occIdx[1][:, None] * voxelSize[1] + POINT_CLOUD_RANGE[1], \
                        occIdx[2][:, None] * voxelSize[2] + POINT_CLOUD_RANGE[2]),
                       dim=1)      # (N, 3) 3: (x, y, z)
    return points, voxel[occIdx], occIdx


def voxel_profile(voxel, voxel_size):
    """
    Args:
        voxel: (N, 3)  3:(x, y, z)
        voxel_size: (vx, vy, vz)

    Returns:
        box: (N, 7) (x, y, z - dz/2, vx, vy, vz, 0)
    """
    centers = torch.cat((voxel[:, :2], voxel[:, 2][:, None] - voxel_size[2] / 2), dim=1)     # (x, y, z - dz/2)
    wlh = torch.cat((torch.tensor(voxel_size[0]).repeat(centers.shape[0])[:, None],
                     torch.tensor(voxel_size[1]).repeat(centers.shape[0])[:, None],
                     torch.tensor(voxel_size[2]).repeat(centers.shape[0])[:, None]), dim=1)
    yaw = torch.full_like(centers[:, 0:1], 0)
    return torch.cat((centers, wlh, yaw), dim=1)


def rotz(t):
    """Rotation about the z-axis."""
    c = torch.cos(t)
    s = torch.sin(t)
    return torch.tensor([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def my_compute_box_3d(center, size, heading_angle):
    """
    Args:
        center: (N, 3)  3: (x, y, z - dz/2)
        size: (N, 3)    3: (vx, vy, vz)
        heading_angle: (N, 1)
    Returns:
        corners_3d: (N, 8, 3)
    """
    h, w, l = size[:, 2], size[:, 0], size[:, 1]
    center[:, 2] = center[:, 2] + h / 2
    l, w, h = (l / 2).unsqueeze(1), (w / 2).unsqueeze(1), (h / 2).unsqueeze(1)
    x_corners = torch.cat([-l, l, l, -l, -l, l, l, -l], dim=1)[..., None]
    y_corners = torch.cat([w, w, -w, -w, w, w, -w, -w], dim=1)[..., None]
    z_corners = torch.cat([h, h, h, h, -h, -h, -h, -h], dim=1)[..., None]
    corners_3d = torch.cat([x_corners, y_corners, z_corners], dim=2)
    corners_3d[..., 0] += center[:, 0:1]
    corners_3d[..., 1] += center[:, 1:2]
    corners_3d[..., 2] += center[:, 2:3]
    return corners_3d


def show_point_cloud(points: np.ndarray, colors=True, points_colors=None, bbox3d=None, voxelize=False,
                     bbox_corners=None, linesets=None, vis=None, offset=[0,0,0], large_voxel=True, voxel_size=0.4):
    """
    :param points: (N, 3)  3:(x, y, z)
    :param colors: false 不显示点云颜色
    :param points_colors: (N, 4）
    :param bbox3d: voxel grid (N, 7) 7: (center, wlh, yaw=0)
    :param voxelize: false 不显示voxel边界
    :param bbox_corners: (N, 8, 3)  voxel grid 角点坐标, 用于绘制voxel grid 边界.
    :param linesets: 用于绘制voxel grid 边界.
    :return:
    """
    if vis is None:
        # Use matplotlib instead of open3d
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        vis = {'fig': fig, 'ax': ax}
    
    if isinstance(offset, list) or isinstance(offset, tuple):
        offset = np.array(offset)
    
    # Plot points
    ax = vis['ax']
    if colors and points_colors is not None:
        ax.scatter(points[:, 0] + offset[0], points[:, 1] + offset[1], points[:, 2] + offset[2], 
                  c=points_colors[:, :3] / 255.0, marker='o', s=10, alpha=0.6)
    else:
        ax.scatter(points[:, 0] + offset[0], points[:, 1] + offset[1], points[:, 2] + offset[2], 
                  c='b', marker='o', s=10, alpha=0.6)
    
    # Plot coordinate frame
    axis_length = 1.0
    ax.quiver(offset[0], offset[1], offset[2], axis_length, 0, 0, color='r', arrow_length_ratio=0.1)
    ax.quiver(offset[0], offset[1], offset[2], 0, axis_length, 0, color='g', arrow_length_ratio=0.1)
    ax.quiver(offset[0], offset[1], offset[2], 0, 0, axis_length, color='b', arrow_length_ratio=0.1)
    
    # Plot voxel edges if needed
    if voxelize and bbox_corners is not None and linesets is not None:
        for i in range(linesets.shape[0]):
            p1 = bbox_corners[linesets[i, 0]] + offset
            p2 = bbox_corners[linesets[i, 1]] + offset
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-', linewidth=1)
    
    return vis


def show_occ(occ_state, occ_show, voxel_size, vis=None, offset=[0, 0, 0]):
    """
    Args:
        occ_state: (Dx, Dy, Dz), cls_id
        occ_show: (Dx, Dy, Dz), bool
        voxel_size: [0.4, 0.4, 0.4]
        vis: Visualizer
        offset:

    Returns:

    """
    colors = colormap_to_colors / 255
    pcd, labels, occIdx = voxel2points(occ_state, occ_show, voxel_size)
    # pcd: (N, 3)  3: (x, y, z)
    # labels: (N, )  cls_id
    _labels = labels % len(colors)
    pcds_colors = colors[_labels]   # (N, 4)

    bboxes = voxel_profile(pcd, voxel_size)    # (N, 7)   7: (x, y, z - dz/2, dx, dy, dz, 0)
    bboxes_corners = my_compute_box_3d(bboxes[:, 0:3], bboxes[:, 3:6], bboxes[:, 6:7])      # (N, 8, 3)

    bases_ = torch.arange(0, bboxes_corners.shape[0] * 8, 8)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])  # lines along y-axis
    edges = edges.reshape((1, 12, 2)).repeat(bboxes_corners.shape[0], 1, 1)     # (N, 12, 2)
    # (N, 12, 2) + (N, 1, 1) --> (N, 12, 2)   此时edges中记录的是bboxes_corners的整体id: (0, N*8).
    edges = edges + bases_[:, None, None]

    vis = show_point_cloud(
        points=pcd.numpy(),
        colors=True,
        points_colors=pcds_colors,
        voxelize=True,
        bbox3d=bboxes.numpy(),
        bbox_corners=bboxes_corners.numpy(),
        linesets=edges.numpy(),
        vis=vis,
        offset=offset,
        large_voxel=True,
        voxel_size=0.4
    )
    return vis


def generate_the_ego_car():
    ego_range = [-2, -1, 0, 2, 1, 1.5]
    ego_voxel_size=[0.1, 0.1, 0.1]
    ego_xdim = int((ego_range[3] - ego_range[0]) / ego_voxel_size[0])
    ego_ydim = int((ego_range[4] - ego_range[1]) / ego_voxel_size[1])
    ego_zdim = int((ego_range[5] - ego_range[2]) / ego_voxel_size[2])
    temp_x = np.arange(ego_xdim)
    temp_y = np.arange(ego_ydim)
    temp_z = np.arange(ego_zdim)
    ego_xyz = np.stack(np.meshgrid(temp_y, temp_x, temp_z), axis=-1).reshape(-1, 3)
    ego_point_x = (ego_xyz[:, 0:1] + 0.5) / ego_xdim * (ego_range[3] - ego_range[0]) + ego_range[0]
    ego_point_y = (ego_xyz[:, 1:2] + 0.5) / ego_ydim * (ego_range[4] - ego_range[1]) + ego_range[1]
    ego_point_z = (ego_xyz[:, 2:3] + 0.5) / ego_zdim * (ego_range[5] - ego_range[2]) + ego_range[2]
    ego_point_xyz = np.concatenate((ego_point_y, ego_point_x, ego_point_z), axis=-1)
    ego_points_label =  (np.ones((ego_point_xyz.shape[0]))*16).astype(np.uint8)
    ego_dict = {}
    ego_dict['point'] = ego_point_xyz
    ego_dict['label'] = ego_points_label
    return ego_point_xyz


def occ2img(semantics=None, is_pano=False, panoptics=None):
    """将占据栅格转换为可视化图像"""
    H, W, D = semantics.shape
    free_id = len(occ_class_names) - 1
    
    # 初始化2D语义图
    semantics_2d = np.ones([H, W], dtype=np.int32) * free_id
    for i in range(D):
        semantics_i = semantics[..., i]
        non_free_mask = (semantics_i != free_id)
        semantics_2d[non_free_mask] = semantics_i[non_free_mask]

    # 应用颜色映射
    viz = color_map[semantics_2d]
    viz = viz[..., :3]

    # 创建实例掩码
    inst_mask = np.zeros_like(semantics_2d).astype(np.bool)
    for ind in inst_class_ids:
        inst_mask[semantics_2d == ind] = True
    
    # 如果是全景分割，应用实例颜色
    if is_pano:
        panoptics_2d = np.ones([H, W], dtype=np.int32) * 0
        for i in range(D):
            panoptics_i = panoptics[..., i]
            semantics_i = semantics[..., i]
            non_free_mask = (semantics_i != free_id)
            panoptics_2d[non_free_mask] = panoptics_i[non_free_mask]
        
        viz_pano = pano_color_map[panoptics_2d]
        viz[inst_mask, :] = viz_pano[inst_mask, :]

    # 调整大小
    viz = cv2.resize(viz, dsize=(800, 800))
    return viz


def visualize(pred_occ, info, vis_dir="results", scale_factor=4, canvas_size=1000, visible=False):
    # prepare save path and medium
    os.makedirs(vis_dir, exist_ok=True)

    views = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ]
    print('start visualizing results')

    # load imgs
    imgs = []
    for view in views:
        img = cv2.imread(info[view]['filename'])
        imgs.append(img)

    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Process occupancy data
    voxel_show = pred_occ != FREE_LABEL
    voxel_size = VOXEL_SIZE
    
    # Get coordinates of occupied voxels
    occupied_indices = np.where(voxel_show)
    x = occupied_indices[0] * voxel_size[0] + POINT_CLOUD_RANGE[0]
    y = occupied_indices[1] * voxel_size[1] + POINT_CLOUD_RANGE[1]
    z = occupied_indices[2] * voxel_size[2] + POINT_CLOUD_RANGE[2]
    
    # Get colors for each occupied voxel
    colors = color_map[pred_occ[occupied_indices]]
    # Normalize colors to [0, 1] for matplotlib
    colors = colors[:, :3] / 255.0

    # Plot voxels with colors
    ax.scatter(x, y, z, c=colors, marker='s', alpha=0.6)

    # Set view angle similar to original
    ax.view_init(elev=20, azim=45)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Save the plot
    plt.savefig(os.path.join(vis_dir, 'occ_3d.png'), bbox_inches='tight', dpi=100)
    plt.close()

    # Create 2D visualization using occ2img function
    occ_2d = occ2img(semantics=pred_occ)
    cv2.imwrite(os.path.join(vis_dir, 'occ_2d.png'), occ_2d[..., ::-1])  # Convert RGB to BGR for OpenCV
    
    # Read the saved 3D image for further processing
    occ_canvas = cv2.imread(os.path.join(vis_dir, 'occ_3d.png'))
    occ_canvas_resize = cv2.resize(occ_canvas, (canvas_size, canvas_size), interpolation=cv2.INTER_CUBIC)

    # Create composite image
    big_img = np.zeros((900 * 2 + canvas_size * scale_factor, 1600 * 3, 3),
                    dtype=np.uint8)
    big_img[:900, :, :] = np.concatenate(imgs[:3], axis=1)
    img_back = np.concatenate(
        [imgs[3][:, ::-1, :], imgs[4][:, ::-1, :], imgs[5][:, ::-1, :]],
        axis=1)
    big_img[900 + canvas_size * scale_factor:, :, :] = img_back
    big_img = cv2.resize(big_img, (int(1600 / scale_factor * 3),
                                    int(900 / scale_factor * 2 + canvas_size)))
    w_begin = int((1600 * 3 / scale_factor - canvas_size) // 2)
    big_img[int(900 / scale_factor):int(900 / scale_factor) + canvas_size,
            w_begin:w_begin + canvas_size, :] = occ_canvas_resize

    # Save images
    for i, img in enumerate(imgs):
        cv2.imwrite(os.path.join(vis_dir, f'img{i}.png'), img)
    cv2.imwrite(os.path.join(vis_dir, 'occ_3d.png'), occ_canvas)
    cv2.imwrite(os.path.join(vis_dir, 'overall.png'), big_img)
    print(f"Saved visualize result to {vis_dir}")
