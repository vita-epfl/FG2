import torch
import torch.nn.functional as F
import configparser
config = configparser.ConfigParser()
config.read("./config.ini")

grd_bev_res = config.getint("VIGOR", "grd_bev_res")
sat_bev_res = config.getint("VIGOR", "sat_bev_res")
grid_size_h = config.getfloat("VIGOR", "grid_size_h")
eps = config.getfloat("Constants", "epsilon")


def compute_vce_loss(X0, Rgt, tgt, R, t):
    """
    Computes Virtual Correspondence Error loss between ground-truth and predicted transformations.

    Args:
        X0 (Tensor): Initial 3D coordinates [B, N, 3].
        Rgt (Tensor): Ground-truth rotation matrix [B, 3, 3].
        tgt (Tensor): Ground-truth translation vector [B, 1, 3].
        R (Tensor): Predicted rotation matrix [B, 3, 3].
        t (Tensor): Predicted translation vector [B, 1, 3].

    Returns:
        loss (Tensor): Mean reprojection error per batch.
    """
    B = X0.shape[0]

    # Transform points using ground-truth and predicted transformations
    X1_gt = Rgt @ X0.repeat(B,1,1).transpose(2, 1) + tgt.transpose(2, 1) 
    X1_pred = R @ X0.repeat(B,1,1).transpose(2, 1) + t.transpose(2, 1) 

    # Compute L2 distance 
    loss = torch.mean(torch.sqrt(((X1_gt - X1_pred)**2).sum(dim=1)), dim=-1)

    return loss


def compute_infonce_loss(Rgt, tgt, matching_score, sampled_idx_sat, sampled_idx_grd, sat_indices_B_seleted, grd_indices_B_seleted):
    
    B, _, _ = matching_score.shape
    
    matches_row = matching_score.flatten(1)
    
    
    T_s2g = torch.zeros(B, 3, 3).to(matching_score.device)
    T_s2g[:,:2,:2] = Rgt
    T_s2g[:,:2,2] = tgt[:,0,:] / grid_size_h * grd_bev_res
    T_s2g[:, 2, 2] = 1
    
    ### satellite to ground
    
    grd_indices_B_mapped = (T_s2g @ sat_indices_B_seleted.transpose(2, 1)).permute(0,2,1)
    grd_indices_B_mapped[:,:,0] = grd_indices_B_mapped[:,:,0] / grd_indices_B_mapped[:,:,2]
    grd_indices_B_mapped[:,:,1] = grd_indices_B_mapped[:,:,1] / grd_indices_B_mapped[:,:,2]
    
    sat_indices_B_seleted = sat_indices_B_seleted[:,:,:2]
    grd_indices_B_mapped = torch.round(grd_indices_B_mapped[:,:,:2])

    
    sat_indices_B_seleted[:,:,0] += (sat_bev_res-1)/2
    sat_indices_B_seleted[:,:,1] += (sat_bev_res-1)/2
    grd_indices_B_mapped[:,:,0] += (grd_bev_res-1)/2
    grd_indices_B_mapped[:,:,1] += (grd_bev_res-1)/2
    
    keep_index = (grd_indices_B_mapped >= 0) & (grd_indices_B_mapped < grd_bev_res)
    keep_index = keep_index[:,:,0] * keep_index[:,:,1]

    matching_row_indices = (sat_indices_B_seleted[:,:,0] * sat_bev_res + sat_indices_B_seleted[:,:,1]).int()
    matching_col_indices = (grd_indices_B_mapped[:,:,0] * grd_bev_res + grd_indices_B_mapped[:,:,1]).int()

    infoNCE_loss_s2g = torch.zeros(B)
    for b in range(B):
        if keep_index[b].sum() > 0:
            selected_indices = torch.nonzero(keep_index[b]).to(matching_score.device)[:,0]
            
            matching_row_indices_selected = torch.index_select(matching_row_indices[b], 0, selected_indices)
            matching_col_indices_selected = torch.index_select(matching_col_indices[b], 0, selected_indices)

            unique, idx, counts = torch.unique(matching_row_indices_selected, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat((torch.tensor([0]).to(matching_score.device), cum_sum[:-1]))
            unique_row_indicies = matching_row_indices_selected[ind_sorted[cum_sum]]
            coorespond_col_indicies = matching_col_indices_selected[ind_sorted[cum_sum]]

            selected_matching_indices = unique_row_indicies*grd_bev_res*grd_bev_res + coorespond_col_indicies
            
            positives = torch.exp(torch.index_select(matches_row[b], 0, selected_matching_indices))

            demoninator = torch.sum(torch.exp(matching_score[b, unique_row_indicies, :]), dim=1)
            infoNCE_loss_s2g[b] = -torch.mean(torch.log(positives / demoninator))

    ### ground to satellite
    
    sat_indices_B_mapped = (torch.linalg.inv(T_s2g) @ grd_indices_B_seleted.transpose(2, 1)).permute(0,2,1)
    sat_indices_B_mapped[:,:,0] = sat_indices_B_mapped[:,:,0] / sat_indices_B_mapped[:,:,2]
    sat_indices_B_mapped[:,:,1] = sat_indices_B_mapped[:,:,1] / sat_indices_B_mapped[:,:,2]
    
    sat_indices_B_mapped = torch.round(sat_indices_B_mapped[:,:,:2])
    grd_indices_B_seleted = grd_indices_B_seleted[:,:,:2] 
    
    sat_indices_B_mapped[:,:,0] += (sat_bev_res-1)/2
    sat_indices_B_mapped[:,:,1] += (sat_bev_res-1)/2
    grd_indices_B_seleted[:,:,0] += (grd_bev_res-1)/2
    grd_indices_B_seleted[:,:,1] += (grd_bev_res-1)/2
    
    keep_index = ((sat_indices_B_mapped[:,:,0] >= 0) & (sat_indices_B_mapped[:,:,0] < sat_bev_res) ) * ((sat_indices_B_mapped[:,:,1] >= 0) & (sat_indices_B_mapped[:,:,1] < sat_bev_res) )
    
    matching_row_indices = (sat_indices_B_mapped[:,:,0] * sat_bev_res + sat_indices_B_mapped[:,:,1]).int()
    matching_col_indices = (grd_indices_B_seleted[:,:,0] * grd_bev_res + grd_indices_B_seleted[:,:,1]).int()

    infoNCE_loss_g2s = torch.zeros(B)
    for b in range(B):
        if keep_index[b].sum() > 0:
            selected_indices = torch.nonzero(keep_index[b]).to(matching_score.device)[:,0]
            
            matching_row_indices_selected = torch.index_select(matching_row_indices[b], 0, selected_indices)
            matching_col_indices_selected = torch.index_select(matching_col_indices[b], 0, selected_indices)

            unique, idx, counts = torch.unique(matching_col_indices_selected, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat((torch.tensor([0]).to(matching_score.device), cum_sum[:-1]))
            unique_col_indicies = matching_col_indices_selected[ind_sorted[cum_sum]]
            coorespond_row_indicies = matching_row_indices_selected[ind_sorted[cum_sum]]

            
            selected_matching_indices = coorespond_row_indicies*grd_bev_res*grd_bev_res + unique_col_indicies
            positives = torch.exp(torch.index_select(matches_row[b], 0, selected_matching_indices))
            demoninator = torch.sum(torch.exp(matching_score[b, :, unique_col_indicies]), dim=0)
            
            infoNCE_loss_g2s[b] = -torch.mean(torch.log(positives / demoninator))
    
    return (infoNCE_loss_s2g + infoNCE_loss_g2s) / 2