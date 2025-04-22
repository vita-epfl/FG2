import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

import argparse
from torch.utils.data import DataLoader, Dataset, Subset
import torch
import torch.nn as nn
import numpy as np
import math
import random

from dataloaders.dataloader_vigor import VIGORDataset, transform_grd, transform_sat
from models.model_vigor import CVM
from models.modules import DinoExtractor
from utils.utils import weighted_procrustes_2d, create_metric_grid, soft_inlier_counting_bev, inlier_counting_bev, e2eProbabilisticProcrustesSolver


parser = argparse.ArgumentParser()
parser.add_argument('-a', '--area', type=str, choices=('samearea','crossarea'), default='samearea')
parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=24)
parser.add_argument('--random_orientation', choices=('True','False'), default='False')
parser.add_argument(
    '--first_run',
    choices=('True', 'False'),
    default='False',
    help='Set to "True" for the first run (infer orientation), or "False" for the second run (use predicted orientation).'
)
parser.add_argument('--ransac', choices=('True','False'), default='False')

args = vars(parser.parse_args())
area = args['area']
batch_size = args['batch_size']
random_orientation = args['random_orientation'] == 'True'
first_run = args['first_run'] == 'True'
ransac = args['ransac'] == 'True'


# Load configuration
import configparser
config = configparser.ConfigParser()
config.read("./config.ini")
import ast

dataset_root = config["VIGOR"]["dataset_root"]
label_root = config["VIGOR"]["label_root"]

grd_bev_res = config.getint("VIGOR", "grd_bev_res")
grd_height_res = config.getint("VIGOR", "grd_height_res")
sat_bev_res = config.getint("VIGOR", "sat_bev_res")
num_samples_matches = config.getint("VIGOR", "num_samples_matches")

ground_image_size = ast.literal_eval(config.get("VIGOR", "ground_image_size"))
satellite_image_size = ast.literal_eval(config.get("VIGOR", "satellite_image_size"))

grid_size_h = config.getfloat("VIGOR", "grid_size_h") 
grid_size_v = config.getfloat("VIGOR", "grid_size_v") 

th_soft_inlier = config.getfloat("VIGOR", "th_soft_inlier") 
th_inlier = config.getfloat("VIGOR", "th_inlier") 
num_samples_matches_ransac = config.getint("VIGOR", "num_samples_matches_ransac")
num_corr_2d_2d = config.getint("VIGOR", "num_corr_2d_2d")
it_matches = config.getint("VIGOR", "it_matches")
it_RANSAC_procrustes = config.getint("VIGOR", "it_RANSAC_procrustes")
num_ref_steps = config.getint("VIGOR", "num_ref_steps")

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
                     transform=(transform_grd, transform_sat), random_orientation=random_orientation, \
                    first_run=first_run)
test_dataloader = DataLoader(vigor, batch_size=batch_size, shuffle=False)


print(f"Area: {area}, Random orientation: {random_orientation}, RANSAC: {ransac}")


torch.cuda.empty_cache()
shared_feature_extractor = DinoExtractor().to(device)

CVM_model = CVM(device, grd_bev_res=grd_bev_res, grd_height_res=grd_height_res, sat_bev_res=sat_bev_res, grid_size_h=grid_size_h, grid_size_v=grid_size_v)

if random_orientation:
    if first_run:
        CVM_model.load_state_dict(torch.load(os.path.join('checkpoints/VIGOR', area,'unknown_ori', 'first_run', 'model.pt')))
        results_dir = os.path.join('results', 'vigor', area, 'unknown_ori', 'first_run')
    else:
        CVM_model.load_state_dict(torch.load(os.path.join('checkpoints/VIGOR', area,'unknown_ori', 'second_run', 'model.pt')))
        results_dir = os.path.join('results', 'vigor', area, 'unknown_ori', 'second_run')

else:
    CVM_model.load_state_dict(torch.load(os.path.join('checkpoints/VIGOR', area,'known_ori', 'model.pt')))
    results_dir = os.path.join('results', 'vigor', area, 'known_ori')
os.makedirs(results_dir, exist_ok=True)


CVM_model.to(device)

# define a metric grid 
metric_coord_sat_B = create_metric_grid(grid_size_h, sat_bev_res, batch_size).to(device)
metric_coord_grd_B = create_metric_grid(grid_size_h, grd_bev_res, batch_size).to(device)

CVM_model.eval()
with torch.no_grad():
    translation_error = []
    yaw_error = []
    yaw_pred_list = []
    yaw_gt_list = []

    for i, data in enumerate(test_dataloader, 0):
        grd, sat, tgt, Rgt, city = data
        B, _, sat_size, _ = sat.size()
        
        grd = grd.to(device)
        sat = sat.to(device)
        tgt = tgt.to(device) 
        Rgt = Rgt.to(device)

        grd_feature = shared_feature_extractor(grd)
        sat_feature = shared_feature_extractor(sat)

        matching_score, matching_score_original = CVM_model(grd_feature, sat_feature)
    
        B, num_kpts_sat, num_kpts_grd = matching_score.shape

        if ransac:
            e2e_Procrustes = e2eProbabilisticProcrustesSolver(it_RANSAC_procrustes, it_matches, num_samples_matches_ransac, num_corr_2d_2d, num_ref_steps, th_inlier, th_soft_inlier, metric_coord_sat_B, metric_coord_grd_B)
            R, t, _, inliers = e2e_Procrustes.estimate_pose(matching_score, return_inliers=False)

        else:
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
    
        if t is None:
            print('t is None')
            yaw_pred_list.append('None')
            yaw_gt_list.append('None')
            continue

        t = t / grid_size_h * sat_size
        
        offset = torch.abs(t-tgt).cpu().detach().numpy()
        translation_error_B = np.sqrt(offset[:,0,0]**2 + offset[:,0,1]**2)

        Rgt = Rgt.cpu().detach().numpy()
        R = R.cpu().detach().numpy()
        for b in range(B):
            if city[b] == 'NewYork':
                translation_error.append(translation_error_B[b] * NewYork_res)
            elif city[b] == 'Seattle':
                translation_error.append(translation_error_B[b] * Seattle_res)
            elif city[b] == 'SanFrancisco':
                translation_error.append(translation_error_B[b] * SanFrancisco_res)
            elif city[b] == 'Chicago':
                translation_error.append(translation_error_B[b] * Chicago_res)
                
            cos = R[b,0,0]
            sin = R[b,1,0]
            yaw = np.degrees( np.arctan2(sin, cos) )            
            
            cos_gt = Rgt[b,0,0]
            sin_gt = Rgt[b,1,0]
            
            yaw_gt = np.degrees( np.arctan2(sin_gt, cos_gt) )
            
            diff = np.abs(yaw - yaw_gt)

            yaw_error.append(np.min([diff, 360-diff]))

            yaw_pred_list.append(str(round(yaw, 2)))
            yaw_gt_list.append(str(round(yaw_gt, 2)))
        

    translation_error_mean = np.mean(translation_error)    
    translation_error_median = np.median(translation_error)
    
    
    yaw_error_mean = np.mean(yaw_error)    
    yaw_error_median = np.median(yaw_error) 

    print('translation_error_mean', translation_error_mean)
    print('translation_error_median', translation_error_median)

    print('yaw_error_mean', yaw_error_mean)
    print('yaw_error_median', yaw_error_median)
    
    with open(os.path.join(results_dir, 'results.txt'), 'w') as f:
        f.write(f'translation_error_mean: {translation_error_mean:.6f}\n')
        f.write(f'translation_error_median: {translation_error_median:.6f}\n')
        f.write(f'yaw_error_mean: {yaw_error_mean:.6f}\n')
        f.write(f'yaw_error_median: {yaw_error_median:.6f}\n')


if random_orientation:
    if first_run:
        with open(os.path.join(results_dir, 'ori_pred.txt'), 'a') as the_file:
            for i in range(len(yaw_pred_list)):
                the_file.write(yaw_pred_list[i]+'\n')
                
        with open(os.path.join(results_dir, 'ori_gt.txt'), 'a') as the_file:
            for i in range(len(yaw_gt_list)):
                the_file.write(yaw_gt_list[i]+'\n')