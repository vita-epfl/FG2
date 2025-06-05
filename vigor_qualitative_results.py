import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

import argparse
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn import functional as F
import torch
import torch.nn as nn
import numpy as np
import math
import random

from dataloaders.dataloader_vigor import VIGORDataset, transform_grd, transform_sat
from models.model_vigor import CVM
from models.modules import DinoExtractor
from utils.utils import weighted_procrustes_2d, create_metric_grid, e2eProbabilisticProcrustesSolver
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--area', type=str, choices=('samearea','crossarea'), default='samearea')
parser.add_argument('--idx', type=int, help='image index', default=0)
parser.add_argument('--match4vis', type=int, help='number of matches to visualize', default=20)

args = vars(parser.parse_args())
area = args['area']
idx = args['idx']
match4vis = args['match4vis']

# Load configuration
import configparser
config = configparser.ConfigParser()
config.read("./config.ini")
import ast

dataset_root = config["VIGOR"]["dataset_root"]
label_root = config["VIGOR"]["label_root"]

ground_image_size = ast.literal_eval(config.get("VIGOR", "ground_image_size"))
satellite_image_size = ast.literal_eval(config.get("VIGOR", "satellite_image_size"))

grid_size_h = config.getfloat("VIGOR", "grid_size_h") 
grid_size_v = config.getfloat("VIGOR", "grid_size_v") 

grd_bev_res = config.getint("Model", "grd_bev_res")
grd_height_res = config.getint("Model", "grd_height_res")
sat_bev_res = config.getint("Model", "sat_bev_res")
num_samples_matches = config.getint("Model", "num_samples_matches")

seed = config.getint("RandomSeed", "seed")
eps = config.getfloat("Constants", "epsilon")

NewYork_res = config.getfloat("Constants", "NewYork_res") * 640 / satellite_image_size[0]
Seattle_res = config.getfloat("Constants", "Seattle_res") * 640 / satellite_image_size[0]
SanFrancisco_res = config.getfloat("Constants", "SanFrancisco_res") * 640 / satellite_image_size[0]
Chicago_res = config.getfloat("Constants", "Chicago_res") * 640 / satellite_image_size[0]


torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(mode=True, warn_only=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"The device is: {}".format(device)

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(mode=True, warn_only=True)


vigor = VIGORDataset(root=dataset_root, label_root=label_root, split=area, train=False, \
                     transform=(transform_grd, transform_sat), random_orientation=False)


torch.cuda.empty_cache()
shared_feature_extractor = DinoExtractor().to(device)

CVM_model = CVM(device, grd_bev_res=grd_bev_res, grd_height_res=grd_height_res, sat_bev_res=sat_bev_res, grid_size_h=grid_size_h, grid_size_v=grid_size_v)
CVM_model.load_state_dict(torch.load(os.path.join('checkpoints/VIGOR', area,'known_ori', 'model.pt')))

results_dir = os.path.join('results', 'vigor', area, 'qualitative')
os.makedirs(results_dir, exist_ok=True)


CVM_model.to(device)

# define a metric grid 
metric_coord_sat_B = create_metric_grid(grid_size_h, sat_bev_res, 1).to(device)
metric_coord_grd_B = create_metric_grid(grid_size_h, grd_bev_res, 1).to(device)\

x_ = np.linspace(-grid_size_h/2, grid_size_h/2, grd_bev_res)
y_ = np.linspace(-grid_size_h/2, grid_size_h/2, grd_bev_res) 
z_ = np.linspace(-grid_size_v/2, grid_size_v/2, grd_height_res)

grd_x, grd_y, grd_z = np.meshgrid(x_, y_, z_, indexing='ij') # 3D voxel grid with given number (horizontal and vertical) cells
phi = np.sign(grd_y+eps) * np.arccos(grd_x / (np.sqrt(grd_x**2 + grd_y**2)+eps)) 
theta = np.arccos(grd_z / (np.sqrt(grd_x**2 + grd_y**2 + grd_z**2)+eps))

phi = 2*np.pi-phi # to align the orientation direction in ground panorama
phi[phi>2*np.pi] -= 2*np.pi

CVM_model.eval()
with torch.no_grad():
    
    grd, sat, tgt, Rgt, city = vigor.__getitem__(idx)
    
    _, grd_size_h, grd_size_w = grd.shape
    _, sat_size, _ = sat.shape
    grd = grd.unsqueeze(0).to(device)
    sat = sat.unsqueeze(0).to(device)

    grd_feature = shared_feature_extractor(grd)
    sat_feature = shared_feature_extractor(sat)

    matching_score, matching_score_original, height_index = CVM_model(grd_feature, sat_feature)

    B, num_kpts_sat, num_kpts_grd = matching_score.shape
    
    matches_row = matching_score.flatten(1)
    batch_idx = torch.tile(torch.arange(B).view(B, 1), [1, num_samples_matches]).reshape(B, num_samples_matches)
    sampled_idx = torch.multinomial(matches_row, num_samples_matches)

    sampled_idx_sat = torch.div(sampled_idx, num_kpts_grd, rounding_mode='trunc')
    sampled_idx_grd = (sampled_idx % num_kpts_grd)

    # # Sample the positions according to the sample ids
    X = metric_coord_sat_B[batch_idx, sampled_idx_sat, :]
    Y = metric_coord_grd_B[batch_idx, sampled_idx_grd, :]
    weights = matches_row[batch_idx, sampled_idx]
    
    R, t, ok_rank = weighted_procrustes_2d(X, Y, use_weights=True, use_mask=True, w=weights) 
    
    t = (t / grid_size_h * sat_size).cpu().detach().numpy()
    tgt = tgt.numpy() 

    translation_error = np.sqrt((t-tgt)[0,0,0]**2 + (t-tgt)[0,0,1]**2)

    if city == 'NewYork':
        translation_error = translation_error * NewYork_res
    elif city == 'Seattle':
        translation_error = translation_error * Seattle_res
    elif city == 'SanFrancisco':
        translation_error = translation_error * SanFrancisco_res
    elif city == 'Chicago':
        translation_error = translation_error * Chicago_res
    
    print(f"Translation error: {translation_error:.3f} meters")
            
    # visualize local matches    
    sampled_idx4vis = torch.argsort(matches_row, descending=True)[0,:match4vis]
    sampled_idx_sat4vis = torch.div(sampled_idx4vis, num_kpts_grd, rounding_mode='trunc')
    sampled_idx_grd4vis = (sampled_idx4vis % num_kpts_grd)
    sampled_height = height_index.flatten()[sampled_idx_grd4vis].cpu().numpy()

    # === Satellite and Ground Keypoint Indices ===
    sat_row = (sampled_idx_sat4vis // sat_bev_res).cpu().numpy()
    sat_col = (sampled_idx_sat4vis % sat_bev_res).cpu().numpy()
    grd_row = (sampled_idx_grd4vis // grd_bev_res).cpu().numpy()
    grd_col = (sampled_idx_grd4vis % grd_bev_res).cpu().numpy()

    # === Map to coordinates in image space ===
    sat_h = sat_row / sat_bev_res * grd_size_h + grd_size_h / sat_bev_res / 2 # put the keypoint at the center of the BEV cell
    sat_w = sat_col / sat_bev_res * grd_size_h + grd_size_h / sat_bev_res / 2
    sat_points = list(zip(sat_w, sat_h))
    
    grd_points = [
        (phi[r, c, h] / (2 * np.pi) * grd_size_w,
         theta[r, c, h] / np.pi * grd_size_h)
        for r, c, h in zip(grd_row, grd_col, sampled_height)
    ]

    # === Prepare Images ===
    grd_to_show = grd[0].permute(1, 2, 0).cpu().numpy()
    sat_resized = F.interpolate(sat, size=(grd_size_h, grd_size_h), mode='bicubic', align_corners=False)
    sat_to_show = sat_resized[0].permute(1, 2, 0).cpu().numpy()
    
    # === Combine Images Side-by-Side ===
    canvas_width = grd_size_w + grd_size_h + 10
    combined_image = np.ones((grd_size_h, canvas_width, 3))
    combined_image[:, :grd_size_w, :] = grd_to_show
    combined_image[:, 10 + grd_size_w:, :] = sat_to_show
    
    # === Plot Matches ===
    fig, ax = plt.subplots(figsize=(15, 60))
    ax.imshow(combined_image)
    x_offset = 10 + grd_size_w
    
    for (x1, y1), (x2, y2) in zip(grd_points, sat_points):
        ax.plot([x1, x2 + x_offset], [y1, y2], marker='o', markersize=2, color='lime', linestyle='-', linewidth=0.8, zorder=0)
    
    # === Plot GT and Prediction ===
    tgt = (tgt / sat_size * grd_size_h)
    t = (t / sat_size * grd_size_h)
    
    ax.scatter(x_offset + grd_size_h / 2 - tgt[0, 1], grd_size_h / 2 - tgt[0, 0],
               s=300, marker='^', facecolor='g', edgecolors='white', label='GT', zorder=1)
    ax.scatter(x_offset + grd_size_h / 2 - t[0, 0, 1], grd_size_h / 2 - t[0, 0, 0],
               s=300, marker='*', facecolor='gold', edgecolors='white', label='Ours', zorder=1)
    
    # === Finalize and Save ===
    ax.legend(loc='upper right', framealpha=0.8, labelcolor='black', prop={'size': 15})
    ax.axis('off')
    plt.savefig(os.path.join(results_dir, f'{area}_{idx}.png'), bbox_inches='tight', pad_inches=0)
    plt.close()


