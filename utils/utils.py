import torch
import torch.nn.functional as F
import numpy as np
import math

def save_metric(file_path, value, label, epoch):
    with open(file_path, 'ab') as f:
        np.savetxt(f, [value], fmt='%4f', header=f'{label}:', comments=f'{epoch}_')   

def desc_l2norm(desc: torch.Tensor) -> torch.Tensor:
    """
    L2-normalize descriptors with shape [N, C] or [N, C, H, W]
    """
    return F.normalize(desc, p=2, dim=1, eps=1e-10)
    

def weighted_procrustes_2d(A, B, w=None, use_weights=True, use_mask=False, eps=1e-16, check_rank=True):

    assert len(A) == len(B)

    if use_weights:
        W1 = torch.abs(w).sum(1, keepdim=True)
        w_norm = (w / (W1 + eps)).unsqueeze(-1)
        A_mean, B_mean = (w_norm * A).sum(1, keepdim=True), (w_norm * B).sum(1, keepdim=True)
        A_c, B_c = A - A_mean, B - B_mean

        H = A_c.transpose(1, 2) @ (w.unsqueeze(-1) * B_c) if use_mask else A_c.transpose(1, 2) @ (w_norm * B_c)
    else:
        A_mean, B_mean = A.mean(1, keepdim=True), B.mean(1, keepdim=True)
        A_c, B_c = A - A_mean, B - B_mean
        H = A_c.transpose(1, 2) @ B_c

    if check_rank and (torch.linalg.matrix_rank(H) == 1).sum() > 0:
        return None, None, False

    U, S, V = torch.svd(H)
    Z = torch.eye(2, device=A.device).unsqueeze(0).repeat(A.shape[0], 1, 1)
    Z[:, -1, -1] = torch.sign(torch.linalg.det(U @ V.transpose(1, 2)))

    R = V @ Z @ U.transpose(1, 2)
    t = B_mean - A_mean @ R.transpose(1, 2)

    return R, t, True

def create_metric_grid(grid_size, res, batch_size):
    x, y = np.linspace(-grid_size/2, grid_size/2, res), np.linspace(-grid_size/2, grid_size/2, res)
    metric_x, metric_y = np.meshgrid(x, y, indexing='ij')
    metric_x, metric_y = torch.tensor(metric_x).flatten().unsqueeze(0).unsqueeze(-1), torch.tensor(metric_y).flatten().unsqueeze(0).unsqueeze(-1)
    metric_coord = torch.cat((metric_x, metric_y), -1).float()
    return metric_coord.repeat(batch_size, 1, 1)

def create_grid_indices(rows, cols):
    """Create homogeneous grid indices centered at (0,0)."""
    row_vals = np.linspace(-(rows - 1) / 2, (rows - 1) / 2, rows)
    col_vals = np.linspace(-(cols - 1) / 2, (cols - 1) / 2, cols)
    row_grid, col_grid = np.meshgrid(row_vals, col_vals, indexing='ij')

    row_tensor = torch.tensor(row_grid, dtype=torch.float32).flatten().unsqueeze(0).unsqueeze(-1)
    col_tensor = torch.tensor(col_grid, dtype=torch.float32).flatten().unsqueeze(0).unsqueeze(-1)
    ones = torch.ones_like(col_tensor)

    return torch.cat((row_tensor, col_tensor, ones), dim=-1)
    
def soft_inlier_counting_bev(X0, X1, R, t, th=50):
    """
    Computes soft inlier count for BEV.
    """
    beta = 5 / th
    X0_to_1 = (R @ X0.transpose(2, 1)).transpose(2, 1) + t
    dist = (((X0_to_1 - X1).pow(2).sum(-1) + 1e-6).sqrt())
    return torch.sigmoid(beta * (th - dist)).sum(-1, keepdim=True)


def inlier_counting_bev(X0, X1, R, t, th=50):
    """
    Computes binary inlier count for BEV.
    """
    X0_to_1 = (R @ X0.transpose(2, 1)).transpose(2, 1) + t
    dist = (((X0_to_1 - X1).pow(2).sum(-1) + 1e-6).sqrt())
    return ((th - dist) >= 0).float()


class e2eProbabilisticProcrustesSolver():
    """
    e2eProbabilisticProcrustesSolver computes the metric relative pose estimation during test time.
    Note that contrary to the training solver, here, the solver only refines the best pose hypothesis.
    Also, parameters are different during training and testing.
    """
    def __init__(self, it_RANSAC, it_matches, num_samples_matches, num_corr_2d_2d, num_refinements, th_inlier, th_soft_inlier, metric_coord_sat_B, metric_coord_grd_B):

        # Populate Procrustes RANSAC parameters
        self.it_RANSAC = it_RANSAC
        self.it_matches = it_matches
        self.num_samples_matches = num_samples_matches
        self.num_corr_2d_2d = num_corr_2d_2d
        self.num_refinements = num_refinements
        self.th_inlier = th_inlier
        self.th_soft_inlier = th_soft_inlier
        self.metric_coord_sat_B = metric_coord_sat_B
        self.metric_coord_grd_B = metric_coord_grd_B

    def estimate_pose(self, matching_score, return_inliers=False):
        '''
            Given 3D coordinates and matching matrices, estimate_pose computes the metric pose between query and reference images.
            args:
                return_inliers: Optional argument that indicates if a list of the inliers should be returned.
        '''
        device = matching_score.device
        matches = matching_score.detach()

        B, num_kpts_sat, num_kpts_grd = matches.shape

        matches_row = matches.reshape(B, num_kpts_sat*num_kpts_grd)
        batch_idx = torch.tile(torch.arange(B).view(B, 1), [1, self.num_samples_matches]).reshape(B, self.num_samples_matches)
        batch_idx_ransac = torch.tile(torch.arange(B).view(B, 1), [1, self.num_corr_2d_2d]).reshape(B, self.num_corr_2d_2d)

        num_valid_h = 0
        Rs = torch.zeros((B, 0, 2, 2)).to(device)
        ts = torch.zeros((B, 0, 1, 2)).to(device)
        scores_ransac = torch.zeros((B, 0)).to(device)

        # Keep track of X and Y correspondences subset
        it_matches_ids = []
        dict_corr = {}

        for i_i in range(self.it_matches):

            try:
                sampled_idx = torch.multinomial(matches_row, self.num_samples_matches)
            except:
                print('[Except Reached]: Invalid matching matrix! ')
                break

            sampled_idx_sat = torch.div(sampled_idx, num_kpts_grd, rounding_mode='trunc')
            sampled_idx_grd = (sampled_idx % num_kpts_grd)

            # # Sample the positions according to the sample ids
            X = self.metric_coord_sat_B[batch_idx, sampled_idx_sat, :]
            Y = self.metric_coord_grd_B[batch_idx, sampled_idx_grd, :]
            
            weights = matches_row[batch_idx, sampled_idx]

            dict_corr[i_i] = {'X': X, 'Y': Y, 'weights': weights}

            for kk in range(self.it_RANSAC):

                sampled_idx_ransac = torch.multinomial(weights, self.num_corr_2d_2d)

                X_k = X[batch_idx_ransac, sampled_idx_ransac, :]
                Y_k = Y[batch_idx_ransac, sampled_idx_ransac, :]
                weights_k = weights[batch_idx_ransac, sampled_idx_ransac]
                
                # get relative pose in grid space
                R, t, ok_rank = weighted_procrustes_2d(X_k, Y_k, use_weights=False)
                # R, t, ok_rank = weighted_procrustes_2d(X_k, Y_k, use_weights=True, use_mask=True, w=weights_k) 

                if not ok_rank:
                    continue

                invalid_t = (torch.isnan(t).any() or torch.isinf(t).any())
                invalid_R = (torch.isnan(R).any() or torch.isinf(R).any())

                if invalid_t or invalid_R:
                    continue

                # Compute hypothesis score
                score_k = soft_inlier_counting_bev(X, Y, R, t, th=self.th_soft_inlier)
                

                Rs = torch.cat([Rs, R.unsqueeze(1)], 1)
                ts = torch.cat([ts, t.unsqueeze(1)], 1)
                scores_ransac = torch.cat([scores_ransac, score_k], 1)
                it_matches_ids.append(i_i)
                num_valid_h += 1

        if num_valid_h > 0:
            max_ind = torch.argmax(scores_ransac, dim=1)
            R = Rs[batch_idx_ransac[:, 0], max_ind]
            t_metric = ts[batch_idx_ransac[:, 0], max_ind]
            best_inliers = scores_ransac[batch_idx_ransac[:, 0], max_ind]

            # Use subset of correspondences that generated the hypothesis with maximum score
            X_best = torch.zeros_like(X)
            Y_best = torch.zeros_like(Y)
            for i_b in range(len(max_ind)):
                X_best[i_b], Y_best[i_b] = dict_corr[it_matches_ids[max_ind[i_b]]]['X'][i_b], dict_corr[it_matches_ids[max_ind[i_b]]]['Y'][i_b]
            inliers_ref = torch.zeros((B, self.num_samples_matches)).to(device)

            # inliers:
            th_ref = self.num_refinements*[self.th_inlier]
            inliers_pre = self.num_corr_2d_2d * torch.ones_like(best_inliers)
            for i_ref in range(len(th_ref)):
                inliers = inlier_counting_bev(X_best, Y_best, R, t_metric, th=th_ref[i_ref])

                do_ref = (inliers.sum(-1) >= self.num_corr_2d_2d) * (inliers.sum(-1) > inliers_pre)
                inliers_pre[do_ref] = inliers.sum(-1)[do_ref]

                # Check whether any refinements need to be done
                if (do_ref.sum().float() == 0.).item():
                    break
                inliers_ref[do_ref] = inliers[do_ref]
                R[do_ref], t_metric[do_ref], _ = weighted_procrustes_2d(X_best[do_ref], Y_best[do_ref],
                                                                     use_weights=True, use_mask=True,
                                                                     check_rank=False,
                                                                     w=inliers_ref[do_ref])
            best_inliers = soft_inlier_counting_bev(X_best, Y_best, R, t_metric, th=self.th_inlier)
        
        else:
            R = torch.zeros((B, 2, 2)).to(matches.device)
            t_metric = torch.zeros((B, 1, 2)).to(matches.device)
            best_inliers = torch.zeros((B)).to(matches.device)
            
        inliers = None
        if return_inliers:
            if num_valid_h > 0:
    
                # Use subset of correspondences that generated the hypothesis with maximum score
                X_best = torch.zeros_like(X)
                Y_best = torch.zeros_like(Y)
                weights_best = torch.zeros_like(weights)
                for i_b in range(len(max_ind)):
                    X_best[i_b], Y_best[i_b] = dict_corr[it_matches_ids[max_ind[i_b]]]['X'][i_b], dict_corr[it_matches_ids[max_ind[i_b]]]['Y'][i_b]
                    weights_best[i_b] = dict_corr[it_matches_ids[max_ind[i_b]]]['weights'][i_b]
    
                # Compute inliers from latest sampled set of correspondences
                inliers_idxs = inlier_counting_bev(X_best, Y_best, R, t_metric, th=self.th_inlier)
                inliers = []
                for idx_b in range(len(inliers_idxs)):
                    X_inliers = X_best[idx_b, inliers_idxs[idx_b]==1.]
                    Y_inliers = Y_best[idx_b, inliers_idxs[idx_b]==1.]
                    score_inliers = weights_best[idx_b, inliers_idxs[idx_b]==1.]
                    order_corr = torch.argsort(score_inliers, descending=True)
                    inliers_b = torch.cat([X_inliers[order_corr], Y_inliers[order_corr], score_inliers[order_corr].unsqueeze(-1)], dim=1)
                    inliers.append(inliers_b)
            
        return R, t_metric, best_inliers, inliers