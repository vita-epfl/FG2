import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

import argparse
import torch
import torch.nn as nn
import numpy as np
import math
import random
from torch.utils.data import DataLoader

from dataloaders.dataloader_kitti import train_set, test1_set, test2_set, get_meter_per_pixel
from models.model_kitti import CVM
from models.modules import DinoExtractor
from utils.utils import weighted_procrustes_2d, create_metric_grid, create_grid_indices, save_metric
from utils.loss import compute_vce_loss, compute_infonce_loss_kitti

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=24)
parser.add_argument('-t', '--train_or_test', type=str, choices=('train','test'))
parser.add_argument('--epoch_to_resume', type=int, help='from which epoch to continue training', default=0)

args = vars(parser.parse_args())
batch_size = args['batch_size']
train_or_test = args['train_or_test']
epoch_to_resume = args['epoch_to_resume']


# Load configuration
import configparser
config = configparser.ConfigParser()
config.read("./config.ini")
import ast


num_thread_workers = config.getint("KITTI", "num_thread_workers")

learning_rate = config.getfloat("KITTI", "learning_rate")

grid_size_h = config.getfloat("KITTI", "grid_size_h") 
grid_size_v = config.getfloat("KITTI", "grid_size_v") 
ground_image_size = ast.literal_eval(config.get("KITTI", "ground_image_size"))

shift_range_lat = config.getfloat("KITTI", "shift_range_lat")
shift_range_lon = config.getfloat("KITTI", "shift_range_lon")
rotation_range = config.getfloat("KITTI", "rotation_range")

grd_bev_res = config.getint("Model", "grd_bev_res")
grd_height_res = config.getint("Model", "grd_height_res")
sat_bev_res = config.getint("Model", "sat_bev_res")
num_samples_matches = config.getint("Model", "num_samples_matches")

beta = config.getfloat("Loss", "beta") 
loss_grid_size = config.getfloat("Loss", "loss_grid_size") 
num_virtual_point = config.getint("Loss", "num_virtual_point")

seed = config.getint("RandomSeed", "seed")
eps = config.getfloat("Constants", "epsilon")

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(mode=True, warn_only=True)


train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=num_thread_workers, drop_last=False)

test1_loader = DataLoader(test1_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                            num_workers=num_thread_workers, drop_last=False)

test2_loader = DataLoader(test2_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                              num_workers=num_thread_workers, drop_last=False)

meter_per_pixel = get_meter_per_pixel()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"The device is: {}".format(device)

torch.cuda.empty_cache()
shared_feature_extractor = DinoExtractor().to(device)
CVM_model = CVM(device, grd_bev_res=grd_bev_res, grd_height_res=grd_height_res, sat_bev_res=sat_bev_res, grid_size_h=grid_size_h, grid_size_v=grid_size_v)

sat_metric_coord_B = create_metric_grid(grid_size_h, sat_bev_res, batch_size).to(device)
grd_metric_coord_B = create_metric_grid(grid_size_h, grd_bev_res, batch_size, only_front=True).to(device)
metric_coord4loss = create_metric_grid(loss_grid_size, num_virtual_point, 1).to(device)
grd_indices_b = create_grid_indices(grd_bev_res, grd_bev_res, only_front=True).to(device)
sat_indices_b = create_grid_indices(sat_bev_res, sat_bev_res).to(device)

print('')
def eval(CVM_model, test_set):
    if test_set == 'test1':
        test_set = test1_set
        test_loader = test1_loader
    elif test_set == 'test2':
        test_set = test2_set
        test_loader = test2_loader
        
    with torch.no_grad():
        CVM_model.eval()
        translation_error = []
        longitudinal_error = []
        lateral_error = []
        yaw_error = []
        
        for i, data in enumerate(test_loader, 0):
            sat, grd, camera_k, tgt, Rgt = data
            B, _, sat_size, _ = sat.size()
            
            grd = grd.to(device)
            sat = sat.to(device)
            camera_k = camera_k.to(device)
            tgt = tgt.to(device) 
            Rgt = Rgt.to(device)
    
            grd_feature = shared_feature_extractor(grd)
            sat_feature = shared_feature_extractor(sat)
    
            matching_score, matching_score_original = CVM_model(grd_feature, sat_feature, camera_k)
    
            _, num_kpts_sat, num_kpts_grd = matching_score.shape
            
            
            matches_row = matching_score.flatten(1)
            batch_idx = torch.tile(torch.arange(B).view(B, 1), [1, num_samples_matches]).reshape(B, num_samples_matches)
            sampled_idx = torch.multinomial(matches_row, num_samples_matches)
    
            sampled_idx_sat = torch.div(sampled_idx, num_kpts_grd, rounding_mode='trunc')
            sampled_idx_grd = (sampled_idx % num_kpts_grd)
    
            # # Sample the positions according to the sample ids
            X = sat_metric_coord_B[batch_idx, sampled_idx_sat, :]
            Y = grd_metric_coord_B[batch_idx, sampled_idx_grd, :]
            weights = matches_row[batch_idx, sampled_idx]
            
            R, t, ok_rank = weighted_procrustes_2d(X, Y, use_weights=True, use_mask=True, w=weights) 

            if t is None:
                print('t is None')
                continue

            # Scale to pixel units
            t   = (t / grid_size_h * sat_size).cpu().detach().numpy()   # [B,1,2] = [y_forward, x_lateral]
            tgt = tgt.cpu().detach().numpy()
            Rgt = Rgt.cpu().detach().numpy()
            R   = R.cpu().detach().numpy()
            
            for b in range(B):
                loc_pred = [sat_size / 2 - t[b, 0, 1], sat_size / 2 - t[b, 0, 0]]
                loc_gt = [sat_size / 2 - tgt[b, 0, 1], sat_size / 2 - tgt[b, 0, 0]]

                distance = np.sqrt((loc_gt[0]-loc_pred[0])**2+(loc_gt[1]-loc_pred[1])**2) * meter_per_pixel
                translation_error.append(distance)
                
                cos = R[b,0,0]
                sin = R[b,1,0]
                yaw = np.arctan2(sin, cos) / np.pi * 180
                
                cos_gt = Rgt[b,0,0]
                sin_gt = Rgt[b,1,0]
                yaw_gt = np.arctan2(sin_gt, cos_gt) / np.pi * 180
                diff = np.abs(yaw - yaw_gt)
                
                yaw_error.append(np.min([diff, 360-diff]))
                
                e = np.array(loc_pred, dtype=float) - np.array(loc_gt, dtype=float)
    
                # (0=up, 90=left)
                theta = np.deg2rad(-yaw_gt)
                
                # Unit vectors tied to GT heading:
                # forward (longitudinal) and left-normal (lateral)
                u_long = np.array([-np.sin(theta),  np.cos(theta)])   # forward
                u_lat  = np.array([-np.cos(theta), -np.sin(theta)])   # left
                
                # Project to get components (in meters)
                err_longitudinal = np.abs(float(e @ u_long) * meter_per_pixel)
                err_lateral      = np.abs(float(e @ u_lat)  * meter_per_pixel)
                
                longitudinal_error.append(err_longitudinal)
                lateral_error.append(err_lateral)
    
        return np.array(translation_error), np.array(yaw_error), np.array(longitudinal_error), np.array(lateral_error)

if train_or_test == 'train':
    label = (
        f"KITTI"
        f"_grd_bev_res_{grd_bev_res}_grd_height_res_{grd_height_res}"
        f"_sat_bev_res_{sat_bev_res}_learning_rate_{learning_rate}"
    )
    print(label)
    
    if epoch_to_resume > 0:
        CVM_model.load_state_dict(torch.load(os.path.join('checkpoints/KITTI', label, str(epoch_to_resume-1), 'model.pt')))
    
    CVM_model.to(device)
    for param in CVM_model.parameters():
        param.requires_grad = True
    
    
    params = [p for p in CVM_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999))
    
    global_step = 0    
    
    for epoch in range(epoch_to_resume, 100):
        CVM_model.train()
        running_loss = 0.0
    
        for i, data in enumerate(train_loader):
            sat, grd, camera_k, tgt, Rgt = data
            
            B, _, sat_size, _ = sat.size()
            tgt = tgt / sat_size * grid_size_h
            
            grd = grd.to(device)
            sat = sat.to(device)
            camera_k = camera_k.to(device)
            tgt = tgt.to(device) 
            Rgt = Rgt.to(device)      

            with torch.no_grad():
                grd_feature = shared_feature_extractor(grd)
                sat_feature = shared_feature_extractor(sat)
    
            # forward + backward + optimize
            matching_score, matching_score_original = CVM_model(grd_feature, sat_feature, camera_k)
            _, num_kpts_sat, num_kpts_grd = matching_score.shape
            
            matches_row = matching_score.flatten(1)
            batch_idx = torch.tile(torch.arange(B).view(B, 1), [1, num_samples_matches]).reshape(B, num_samples_matches)
            sampled_idx = torch.multinomial(matches_row, num_samples_matches)
    
            sampled_idx_sat = torch.div(sampled_idx, num_kpts_grd, rounding_mode='trunc')
            sampled_idx_grd = (sampled_idx % num_kpts_grd)
    
            # # Sample the positions according to the sample ids
            X = sat_metric_coord_B[batch_idx, sampled_idx_sat, :]
            Y = grd_metric_coord_B[batch_idx, sampled_idx_grd, :]
            weights = matches_row[batch_idx, sampled_idx]
            
            
            R, t, ok_rank = weighted_procrustes_2d(X, Y, use_weights=True, use_mask=True, w=weights) 
            
            if t is None:
                print('t is None')
                continue

            sat_indices_B = sat_indices_b.repeat(B,1,1)
            sat_indices_B_seleted = sat_indices_B[batch_idx,  sampled_idx_sat, :]
            grd_indices_B = grd_indices_b.repeat(B,1,1)
            grd_indices_B_seleted = grd_indices_B[batch_idx, sampled_idx_grd, :]
    
    
            # loss_infonce = compute_infonce_loss_kitti(Rgt, tgt, matching_score_original, sat_indices_B_seleted, grd_indices_B_seleted)
            loss_vce = compute_vce_loss(metric_coord4loss, Rgt, tgt, R, t)
            avg_loss = loss_vce.mean()
        
            # avg_loss = beta * loss_infonce.mean() + loss_vce.mean()
    
            if global_step % 100 == 0:
                print(f'Epoch [{epoch}] Step [{global_step}] Loss: {avg_loss.item():.4f}')
    
            optimizer.zero_grad()
            avg_loss.backward()
            optimizer.step()
            global_step += 1

    
        # Save model
        model_dir = f'checkpoints/KITTI/{label}/{epoch}/'
        os.makedirs(model_dir, exist_ok=True)
        print(f'Saving checkpoint at {model_dir}')
        torch.save(CVM_model.cpu().state_dict(), model_dir + 'model.pt')
        CVM_model.to(device)
    
        # Evaluation
        print('Evaluating...')

        results_dir = f'results/kitti/{label}/'
        os.makedirs(results_dir, exist_ok=True)

        for test_set in ['test1', 'test2']:
            translation_error, yaw_error, _, _ = eval(CVM_model, test_set)
            trans_mean, trans_median = np.mean(translation_error), np.median(translation_error)
            yaw_mean, yaw_median = np.mean(yaw_error), np.median(yaw_error)
    
            print(f'Epoch {epoch} {test_set}: Mean translation error: {trans_mean:.2f}, Median: {trans_median:.2f}')
            print(f'Epoch {epoch} {test_set}: Mean yaw error: {yaw_mean:.2f}, Median: {yaw_median:.2f}')
        
            save_metric(results_dir + 'Mean_distance_error.txt', trans_mean, test_set + '_mean_distance_error', epoch)
            save_metric(results_dir + 'Median_distance_error.txt', trans_median, test_set +'_median_distance_error', epoch)
            save_metric(results_dir + 'Mean_orientation_error.txt', yaw_mean, test_set +'_mean_yaw_error', epoch)
            save_metric(results_dir + 'Median_orientation_error.txt', yaw_median, test_set +'_median_yaw_error', epoch)
        
if train_or_test == 'test':
    CVM_model.load_state_dict(torch.load(os.path.join('checkpoints/KITTI', 'model.pt')))
    CVM_model.to(device)
    
    results_dir = os.path.join('results', 'kitti')
    os.makedirs(results_dir, exist_ok=True)

    for test_set in ['test1', 'test2']:
        translation_error, yaw_error, longitudinal_error, lateral_error = eval(CVM_model, test_set)
        trans_mean, trans_median = np.mean(translation_error), np.median(translation_error)
        yaw_mean, yaw_median = np.mean(yaw_error), np.median(yaw_error)

        with open(os.path.join(results_dir, 'results.txt'), 'a') as f:
            f.write(f'{test_set}: translation error mean (m): {trans_mean:.6f}\n')
            f.write(f'{test_set}: translation error median (m): {trans_median:.6f}\n')
            f.write(f'{test_set}: yaw error mean: {yaw_mean:.6f}\n')
            f.write(f'{test_set}: yaw error median: {yaw_median:.6f}\n')

            # Compute percentages
            perc_lateral_1m = np.mean(lateral_error < 1) * 100
            perc_lateral_3m = np.mean(lateral_error < 3) * 100
            perc_lateral_5m = np.mean(lateral_error < 5) * 100
        
            perc_long_1m = np.mean(longitudinal_error < 1) * 100
            perc_long_3m = np.mean(longitudinal_error < 3) * 100
            perc_long_5m = np.mean(longitudinal_error < 5) * 100
        
            perc_orient_1deg = np.mean(yaw_error < 1) * 100
            perc_orient_3deg = np.mean(yaw_error < 3) * 100
            perc_orient_5deg = np.mean(yaw_error < 5) * 100
        
            # Write to file
            f.write(f'{test_set}: Lateral error <1m: {perc_lateral_1m:.2f}%\n')
            f.write(f'{test_set}: Lateral error <3m: {perc_lateral_3m:.2f}%\n')
            f.write(f'{test_set}: Lateral error <5m: {perc_lateral_5m:.2f}%\n')
        
            f.write(f'{test_set}: Longitudinal error <1m: {perc_long_1m:.2f}%\n')
            f.write(f'{test_set}: Longitudinal error <3m: {perc_long_3m:.2f}%\n')
            f.write(f'{test_set}: Longitudinal error <5m: {perc_long_5m:.2f}%\n')
        
            f.write(f'{test_set}: Orientation error <1°: {perc_orient_1deg:.2f}%\n')
            f.write(f'{test_set}: Orientation error <3°: {perc_orient_3deg:.2f}%\n')
            f.write(f'{test_set}: Orientation error <5°: {perc_orient_5deg:.2f}%\n')