import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

import argparse
from torch.utils.data import DataLoader, Dataset, Subset
import torch
import torch.nn as nn
import numpy as np
import math
import random
import time

from dataloaders.dataloader_vigor import VIGORDataset, transform_grd, transform_sat
from models.model_vigor import CVM
from models.modules import DinoExtractor
from utils.utils import weighted_procrustes_2d, create_metric_grid, create_grid_indices, save_metric
from utils.loss import compute_vce_loss, compute_infonce_loss

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--area', type=str, choices=('samearea','crossarea'), default='samearea')
parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=24)
parser.add_argument('--random_orientation', choices=('True','False'), default='False')

args = vars(parser.parse_args())
area = args['area']
batch_size = args['batch_size']
random_orientation = args['random_orientation'] == 'True'


# Load configuration
import configparser
config = configparser.ConfigParser()
config.read("./config.ini")
import ast

dataset_root = config["VIGOR"]["dataset_root"]
label_root = config["VIGOR"]["label_root"]

learning_rate = config.getfloat("VIGOR", "learning_rate")
epoch_to_resume = config.getint("VIGOR", "epoch_to_resume")
beta = config.getfloat("VIGOR", "beta") 

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


vigor = VIGORDataset(root=dataset_root, label_root=label_root, split=area, train=True, \
                     transform=(transform_grd, transform_sat), random_orientation=random_orientation, \
                    )

dataset_length = int(vigor.__len__())
index_list = np.arange(vigor.__len__())
np.random.shuffle(index_list)
train_indices = index_list[0: int(len(index_list)*0.8)]
val_indices = index_list[int(len(index_list)*0.8):]
training_set = Subset(vigor, train_indices)
val_set = Subset(vigor, val_indices)

train_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

print(f"Area: {area}, Random orientation: {random_orientation}")


label = (
    f"{area}_random_ori_{random_orientation}"
    f"_grd_bev_res_{grd_bev_res}_grd_height_res_{grd_height_res}"
    f"_sat_bev_res_{sat_bev_res}_learning_rate_{learning_rate}"
)
print(label)

torch.cuda.empty_cache()
shared_feature_extractor = DinoExtractor().to(device)
CVM_model = CVM(device, grd_bev_res=grd_bev_res, grd_height_res=grd_height_res, sat_bev_res=sat_bev_res, grid_size_h=grid_size_h, grid_size_v=grid_size_v)

if epoch_to_resume > 0:
    CVM_model.load_state_dict(torch.load(os.path.join('checkpoints/VIGOR', label, str(epoch_to_resume-1), 'model.pt')))
CVM_model.to(device)
for param in CVM_model.parameters():
    param.requires_grad = True


params = [p for p in CVM_model.parameters() if p.requires_grad]

optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999))


global_step = 0


# define a metric grid 
metric_coord_sat_B = create_metric_grid(grid_size_h, sat_bev_res, batch_size).to(device)
metric_coord_grd_B = create_metric_grid(grid_size_h, grd_bev_res, batch_size).to(device)
metric_coord4loss = create_metric_grid(grid_size_h, grd_bev_res, 1).to(device)
grd_indices_b = create_grid_indices(grd_bev_res, grd_bev_res).to(device)
sat_indices_b = create_grid_indices(sat_bev_res, sat_bev_res).to(device)

for epoch in range(epoch_to_resume, 100):
    CVM_model.train()
    running_loss = 0.0

    for i, data in enumerate(train_dataloader):
        grd, sat, tgt, Rgt, city = data
        B, _, sat_size, _ = sat.size()
        tgt = tgt / sat_size * grid_size_h

        grd, sat, tgt, Rgt = map(lambda x: x.to(device), [grd, sat, tgt, Rgt])

        # Forward pass
        with torch.no_grad():
            grd_feat = shared_feature_extractor(grd)
            sat_feat = shared_feature_extractor(sat)
        matching_score, score_orig = CVM_model(grd_feat, sat_feat)

        _, num_kpts_sat, num_kpts_grd = matching_score.shape
        matches_flat = matching_score.flatten(1)
        batch_idx = torch.arange(B).view(B, 1).repeat(1, num_samples_matches).reshape(B, num_samples_matches)
        sampled_idx = torch.multinomial(matches_flat, num_samples_matches)

        sampled_sat_idx = torch.div(sampled_idx, num_kpts_grd, rounding_mode='trunc')
        sampled_grd_idx = sampled_idx % num_kpts_grd

        X = metric_coord_sat_B[batch_idx, sampled_sat_idx, :]
        Y = metric_coord_grd_B[batch_idx, sampled_grd_idx, :]
        weights = matches_flat[batch_idx, sampled_idx]

        R, t, ok_rank = weighted_procrustes_2d(X, Y, use_weights=True, use_mask=True, w=weights)

        if t is None:
            print('t is None')
            continue

        loss_vce = compute_vce_loss(metric_coord4loss, Rgt, tgt, R, t)

        sat_idx_B = sat_indices_b.repeat(B, 1, 1)
        grd_idx_B = grd_indices_b.repeat(B, 1, 1)

        sat_selected = sat_idx_B[batch_idx, sampled_sat_idx, :]
        grd_selected = grd_idx_B[batch_idx, sampled_grd_idx, :]

        loss_infonce = compute_infonce_loss(Rgt, tgt, score_orig, sampled_sat_idx, sampled_grd_idx, sat_selected, grd_selected)
        avg_loss = beta * loss_infonce.mean() + loss_vce.mean()
        

        if global_step % 100 == 0:
            print(f'Epoch [{epoch}] Step [{global_step}] Loss: {avg_loss.item():.4f}')

        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()

        global_step += 1

    # Save model
    model_dir = f'checkpoints/VIGOR/{label}/{epoch}/'
    os.makedirs(model_dir, exist_ok=True)
    print(f'Saving checkpoint at {model_dir}')
    torch.save(CVM_model.cpu().state_dict(), model_dir + 'model.pt')
    CVM_model.to(device)

    # Evaluation
    print('Evaluating on validation set...')
    CVM_model.eval()
    translation_errors = []
    yaw_errors = []

    with torch.no_grad():
        for data in val_dataloader:
            grd, sat, tgt, Rgt, city = data
            B, _, sat_size, _ = sat.size()
            grd, sat, tgt, Rgt = map(lambda x: x.to(device), [grd, sat, tgt, Rgt])

            grd_feat = shared_feature_extractor(grd)
            sat_feat = shared_feature_extractor(sat)
            matching_score, _ = CVM_model(grd_feat, sat_feat)

            matches_flat = matching_score.flatten(1)
            batch_idx = torch.arange(B).view(B, 1).repeat(1, num_samples_matches).reshape(B, num_samples_matches)
            sampled_idx = torch.multinomial(matches_flat, num_samples_matches)

            sampled_sat_idx = torch.div(sampled_idx, num_kpts_grd, rounding_mode='trunc')
            sampled_grd_idx = sampled_idx % num_kpts_grd

            X = metric_coord_sat_B[batch_idx, sampled_sat_idx, :]
            Y = metric_coord_grd_B[batch_idx, sampled_grd_idx, :]
            weights = matches_flat[batch_idx, sampled_idx]

            R, t, ok_rank = weighted_procrustes_2d(X, Y, use_weights=True, use_mask=True, w=weights)
            if t is None:
                print('t is None (validation)')
                continue

            t = t / grid_size_h * sat_size
            offset = (t - tgt).abs().cpu().numpy()
            trans_error = np.sqrt(offset[:, 0, 0] ** 2 + offset[:, 0, 1] ** 2)
            translation_errors.extend(trans_error)

            Rgt_np, R_np = Rgt.cpu().numpy(), R.cpu().numpy()
            for b in range(B):
                cos = R_np[b,0,0]
                sin = R_np[b,1,0]
                yaw_pred = np.degrees( np.arctan2(sin, cos) )            
                
                cos_gt = Rgt_np[b,0,0]
                sin_gt = Rgt_np[b,1,0]
                
                yaw_gt = np.degrees( np.arctan2(sin_gt, cos_gt) )

                if yaw_pred is None or yaw_gt is None:
                    print('Invalid yaw angle')
                    continue
                yaw_diff = abs(yaw_pred - yaw_gt)
                yaw_errors.append(min(yaw_diff, 360 - yaw_diff))

    # Compute and log metrics
    results_dir = f'results/vigor/{label}/'
    os.makedirs(results_dir, exist_ok=True)

    trans_mean, trans_median = np.mean(translation_errors), np.median(translation_errors)
    yaw_mean, yaw_median = np.mean(yaw_errors), np.median(yaw_errors)

    print(f'Epoch {epoch}: Mean translation error: {trans_mean:.2f}, Median: {trans_median:.2f}')
    print(f'Epoch {epoch}: Mean yaw error: {yaw_mean:.2f}, Median: {yaw_median:.2f}')

    save_metric(results_dir + 'Mean_distance_error.txt', trans_mean, 'Validation_set_mean_distance_error_in_pixels', epoch)
    save_metric(results_dir + 'Median_distance_error.txt', trans_median, 'Validation_set_median_distance_error_in_pixels', epoch)
    save_metric(results_dir + 'Mean_orientation_error.txt', yaw_mean, 'Validation_set_mean_yaw_error', epoch)
    save_metric(results_dir + 'Median_orientation_error.txt', yaw_median, 'Validation_set_median_yaw_error', epoch)

