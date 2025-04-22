import torch
import torch.nn.functional as F
import configparser
config = configparser.ConfigParser()
config.read("./config.ini")

grd_bev_res = config.getint("Model", "grd_bev_res")
sat_bev_res = config.getint("Model", "sat_bev_res")
num_keypoints = config.getint("Model", "num_keypoints")

dataset = config["Dataset"]["dataset"]
if dataset == 'VIGOR':
    grid_size_h = config.getfloat("VIGOR", "grid_size_h")
elif dataset == 'RobotCar':
    grid_size_h = config.getfloat("RobotCar", "grid_size_h")
    

def loss_bev_space(X0, Rgt, tgt, R, t):
    """
    Computes BEV reprojection loss between ground-truth and predicted transformations.

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

    # Compute L2 distance for reprojection error
    loss = torch.mean(torch.sqrt(((X1_gt - X1_pred)**2).sum(dim=1)), dim=-1)

    return loss


def trans_l1_loss(t, tgt):
    """
    Computes L1 loss for translation vector.

    Args:
        t (Tensor): Predicted translation vector [B, 1, 3].
        tgt (Tensor): Ground-truth translation vector [B, 1, 3].

    Returns:
        loss (Tensor): L1 loss for translation.
    """
    return torch.abs(t - tgt).sum(dim=-1)


def trans_l2_loss(t, tgt):
    """
    Computes L2 loss for translation vector.

    Args:
        t (Tensor): Predicted translation vector [B, 1, 3].
        tgt (Tensor): Ground-truth translation vector [B, 1, 3].

    Returns:
        loss (Tensor): L2 loss for translation.
    """
    return ((t[:, :, :2] - tgt[:, :, :2]) ** 2).sum(dim=-1)


def rot_angle_loss(R, Rgt):
    """
    Computes rotation loss using residual rotation angle [radians].

    Args:
        R (Tensor): Predicted rotation matrix [B, 3, 3].
        Rgt (Tensor): Ground-truth rotation matrix [B, 3, 3].

    Returns:
        loss (Tensor): Rotation error (L1 loss).
        R_err (Tensor): Rotation error in radians.
    """
    residual = R.transpose(1, 2) @ Rgt
    trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
    cosine = torch.clip((trace - 1) / 2, -0.99999, 0.99999)  # Prevent NaNs
    R_err = torch.acos(cosine)
    return R_err.unsqueeze(-1), R_err


def compute_pose_loss(R, t, Rgt, tgt, soft_clipping=True):
    """
    Computes total pose estimation loss (rotation + translation).

    Args:
        R (Tensor): Predicted rotation matrix [B, 3, 3].
        t (Tensor): Predicted translation vector [B, 1, 3].
        Rgt (Tensor): Ground-truth rotation matrix [B, 3, 3].
        tgt (Tensor): Ground-truth translation vector [B, 1, 3].
        soft_clipping (bool): Whether to apply soft clipping using tanh.

    Returns:
        loss (Tensor): Combined loss.
        loss_rot (Tensor): Rotation loss.
        loss_trans (Tensor): Translation loss.
    """
    loss_rot, rot_err = rot_angle_loss(R, Rgt)
    loss_trans = trans_l1_loss(t, tgt)

    if soft_clipping:
        loss = torch.tanh(loss_rot / 0.9) + torch.tanh(loss_trans / 0.9)
    else:
        loss = loss_rot + loss_trans

    return loss, loss_rot, loss_trans


def vcre_loss(R, t, Tgt, K0, H=720):
    """
    Computes Virtual Correspondences Reprojection Error (VCRE).

    Args:
        R (Tensor): Predicted rotation matrix [B, 3, 3].
        t (Tensor): Predicted translation vector [B, 1, 3].
        Tgt (Tensor): Ground-truth transformation matrix [B, 4, 4].
        K0 (Tensor): Intrinsic camera matrix.
        H (int): Image height.

    Returns:
        repr_err (Tensor): Reprojection error per batch.
    """
    B = R.shape[0]
    Rgt, tgt = Tgt[:, :3, :3], Tgt[:, :3, 3:].transpose(1, 2)

    eye_coords = torch.from_numpy(eye_coords_glob).to(R.device, dtype=torch.float32).unsqueeze(0)[:, :, :3]
    eye_coords = eye_coords.expand(B, -1, -1)

    # Ground-truth 2D projections
    uv_gt = project_2d(eye_coords, K0)

    # Predicted transformations
    eye_coord_tmp = (R @ eye_coords.transpose(2, 1)) + t.transpose(2, 1)
    eyes_residual = (Rgt.transpose(2, 1) @ eye_coord_tmp - Rgt.transpose(2, 1) @ tgt.transpose(2, 1)).transpose(2, 1)

    uv_pred = project_2d(eyes_residual, K0)

    # Clip values to prevent invalid pixel locations
    uv_gt = torch.clip(uv_gt, 0, H)
    uv_pred = torch.clip(uv_pred, 0, H)

    # Compute reprojection error
    repr_err = torch.mean(torch.norm(uv_gt - uv_pred, dim=-1), dim=-1, keepdim=True)

    return repr_err


def compute_vcre_loss(R, t, Rgt, tgt, K=None, soft_clipping=True):
    """
    Computes Virtual Correspondences Reprojection Error (VCRE) loss.

    Args:
        R (Tensor): Predicted rotation matrix [B, 3, 3].
        t (Tensor): Predicted translation vector [B, 1, 3].
        Rgt (Tensor): Ground-truth rotation matrix [B, 3, 3].
        tgt (Tensor): Ground-truth translation vector [B, 1, 3].
        K (Tensor, optional): Camera intrinsic matrix.
        soft_clipping (bool): Whether to apply soft clipping using tanh.

    Returns:
        loss (Tensor): VCRE loss.
        loss_rot (Tensor): Rotation loss.
        loss_trans (Tensor): Translation loss.
    """
    B = R.shape[0]
    Tgt = torch.zeros((B, 4, 4), device=R.device, dtype=torch.float32)
    Tgt[:, :3, :3] = Rgt
    Tgt[:, :3, 3:] = tgt.transpose(2, 1)

    loss = vcre_loss(R, t, Tgt, K)

    if soft_clipping:
        loss = torch.tanh(loss / 80)

    loss_rot, rot_err = rot_angle_loss(R, Rgt)
    loss_trans = trans_l1_loss(t, tgt)

    return loss, loss_rot, loss_trans

def compute_similarity_loss(Rgt, tgt, sat_desc, grd_desc, sat_points_selected, grd_points_selected, sat_indices_sampled, grd_indices_sampled, coord_sat, coord_grd):
    """
    Compute similarity loss between satellite and ground features after transformation.

    Args:
        Rgt (Tensor): Ground-truth rotation matrix (B, 2, 2).
        tgt (Tensor): Ground-truth translation vector (B, 1, 2).
        sat_desc (Tensor): Satellite descriptors (B, C, N).
        grd_desc (Tensor): Ground descriptors (B, C, M).
        sat_points_selected (Tensor): Selected satellite points (B, P, 2).
        grd_points_selected (Tensor): Selected ground points (B, P, 2).
        sat_indices_sampled (Tensor): Sampled satellite indices (B, P).
        grd_indices_sampled (Tensor): Sampled ground indices (B, P).
        coord_sat (Tensor): Sampled ground indices (B, sat_bev_res*sat_bev_res, 2).
        coord_grd (Tensor): Sampled ground indices (B, grd_bev_res*grd_bev_res, 2).

    Returns:
        similarity_loss (Tensor): Similarity loss value per batch (B,).
    """
    
    B, num_points, _ = sat_points_selected.shape
    device = sat_desc.device  # Ensure everything is on the correct device

    # Add homogeneous coordinates (convert 2D to 3D)
    ones = torch.ones(B, num_points, 1, device=device)
    sat_points_h = torch.cat((sat_points_selected, ones), dim=-1)  # (B, P, 3)
    grd_points_h = torch.cat((grd_points_selected, ones), dim=-1)  # (B, P, 3)

    # Construct transformation matrix (B, 3, 3)
    T_s2g = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
    T_s2g[:, :2, :2] = Rgt
    T_s2g[:, :2, 2] = tgt[:, 0, :]

    # ------------------------
    # Satellite to Ground Mapping
    # ------------------------
    grd_points_mapped = (T_s2g @ sat_points_h.transpose(2, 1)).permute(0, 2, 1)
    grd_points_mapped[..., :2] /= grd_points_mapped[..., 2:3]  # Normalize by homogeneous coord
    grd_points_mapped = grd_points_mapped[..., :2]  # (B, P, 2)

    # Find nearest ground metric coordinates
    distances = torch.cdist(grd_points_mapped, coord_grd)  # (B, P, R)
    min_distances, grd_indices_mapped = distances.min(dim=-1)  # (B, P)

    keep_index = min_distances <= (grid_size_h / grd_bev_res)  # Mask valid matches

    # Compute similarity loss for Satellite-to-Ground (S2G)
    similarity_loss_s2g = torch.zeros(B, device=device)
    for b in range(B):
        if keep_index[b].sum() > 0:
            sat_indices_kept = sat_indices_sampled[b, keep_index[b]]
            grd_indices_kept = grd_indices_mapped[b, keep_index[b]]

            desc_similarity = torch.matmul(sat_desc[b, :, sat_indices_kept].T, grd_desc[b, :, grd_indices_kept])
            similarity_loss_s2g[b] = keep_index[b].sum() - torch.diagonal(desc_similarity).sum()

    # print('sat_points_selected', sat_points_selected)
    # print('grd_points_mapped', grd_points_mapped)
    # print('coord_grd[0,grd_indices_mapped[0,0],:]', coord_grd[0,grd_indices_mapped[0,0],:])
    
    # ------------------------
    # Ground to Satellite Mapping
    # ------------------------
    T_g2s = torch.linalg.inv(T_s2g)  # Inverse transformation (Ground -> Satellite)
    sat_points_mapped = (T_g2s @ grd_points_h.transpose(2, 1)).permute(0, 2, 1)
    sat_points_mapped[..., :2] /= sat_points_mapped[..., 2:3]  # Normalize by homogeneous coord
    sat_points_mapped = sat_points_mapped[..., :2]  # (B, P, 2)
    
    # Find nearest satellite metric coordinates
    distances = torch.cdist(sat_points_mapped, coord_sat)  # (B, P, R)
    min_distances, sat_indices_mapped = distances.min(dim=-1)  # (B, P)
    keep_index = min_distances <= (grid_size_h / sat_bev_res)  # Mask valid matches
    
    # print('grd_points_selected', grd_points_selected)
    # print('sat_points_mapped', sat_points_mapped)
    # print('coord_sat[0,sat_indices_mapped[0,0],:]', coord_sat[0,sat_indices_mapped[0,0],:])

    # Compute similarity loss for Ground-to-Satellite (G2S)
    similarity_loss_g2s = torch.zeros(B, device=device)
    for b in range(B):
        if keep_index[b].sum() > 0:
            sat_indices_kept = sat_indices_mapped[b, keep_index[b]]
            grd_indices_kept = grd_indices_sampled[b, keep_index[b]]

            desc_similarity = torch.matmul(sat_desc[b, :, sat_indices_kept].T, grd_desc[b, :, grd_indices_kept])
            similarity_loss_g2s[b] = keep_index[b].sum() - torch.diagonal(desc_similarity).sum()

    # Return average loss
    # return (similarity_loss_s2g + similarity_loss_g2s) / 2
    return similarity_loss_s2g


def compute_infonce_loss(
    Rgt, tgt,
    sat_points_selected, grd_points_selected,
    sampled_row, sampled_col,
    sat_indices_topk, grd_indices_topk,
    sat_keypoint_coord, grd_keypoint_coord,
    matching_score_original
):
    """
    Compute InfoNCE loss between satellite and ground features after transformation.

    Args:
        Rgt (Tensor): Ground-truth rotation matrix (B, 2, 2).
        tgt (Tensor): Ground-truth translation vector (B, 1, 2).
        sat_points_selected (Tensor): Selected satellite points (B, P, 2).
        grd_points_selected (Tensor): Selected ground points (B, P, 2).
        sampled_row (Tensor): Ground descriptor row indices for sampled points (B, P).
        sampled_col (Tensor): Satellite descriptor column indices for sampled points (B, P).
        sat_indices_topk, grd_indices_topk: Not used, but assumed available for expansion.
        sat_keypoint_coord (Tensor): Satellite keypoint coordinates (B, R, 2).
        grd_keypoint_coord (Tensor): Ground keypoint coordinates (B, R, 2).
        matching_score_original (Tensor): Original matching score map (B, M, N).

    Returns:
        Tensor: InfoNCE similarity loss (B,)
    """
    B, P, _ = sat_points_selected.shape
    device = matching_score_original.device

    # Homogeneous coordinates
    ones = torch.ones(B, P, 1, device=device)
    sat_points_h = torch.cat([sat_points_selected, ones], dim=-1)  # (B, P, 3)
    grd_points_h = torch.cat([grd_points_selected, ones], dim=-1)  # (B, P, 3)

    # Satellite to Ground Transformation
    T_s2g = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
    T_s2g[:, :2, :2] = Rgt
    T_s2g[:, :2, 2] = tgt[:, 0, :]

    # ------------------------
    # Satellite → Ground
    # ------------------------
    grd_points_mapped = (T_s2g @ sat_points_h.transpose(2, 1)).permute(0, 2, 1)
    grd_points_mapped[..., :2] /= grd_points_mapped[..., 2:3]
    grd_points_mapped = grd_points_mapped[..., :2]

    distances = torch.cdist(grd_points_mapped, grd_keypoint_coord)  # (B, P, R)
    min_distances, col_mapped = distances.min(dim=-1)               # (B, P)
    keep_index = min_distances <= (grid_size_h / grd_bev_res)

    infoNCE_loss_s2g = torch.zeros(B, device=device)

    for b in range(B):
        if keep_index[b].sum() > 0:
            row_kept = sampled_row[b, keep_index[b]]
            col_kept = col_mapped[b, keep_index[b]]

            unique, idx, counts = torch.unique(row_kept, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat([torch.tensor([0], device=device), cum_sum[:-1]])
            unique_row = row_kept[ind_sorted[cum_sum]]
            coorespond_col = col_kept[ind_sorted[cum_sum]]

            selected_indices = unique_row * num_keypoints + coorespond_col
            positives = torch.exp(torch.index_select(matching_score_original[b].flatten(), 0, selected_indices))
            denominator = torch.sum(torch.exp(matching_score_original[b, unique_row, :]), dim=1)

            infoNCE_loss_s2g[b] = -torch.mean(torch.log(positives / denominator))

    # ------------------------
    # Ground → Satellite
    # ------------------------
    T_g2s = torch.linalg.inv(T_s2g)
    sat_points_mapped = (T_g2s @ grd_points_h.transpose(2, 1)).permute(0, 2, 1)
    sat_points_mapped[..., :2] /= sat_points_mapped[..., 2:3]
    sat_points_mapped = sat_points_mapped[..., :2]

    distances = torch.cdist(sat_points_mapped, sat_keypoint_coord)
    min_distances, row_mapped = distances.min(dim=-1)
    keep_index = min_distances <= (grid_size_h / sat_bev_res)

    infoNCE_loss_g2s = torch.zeros(B, device=device)

    for b in range(B):
        if keep_index[b].sum() > 0:
            row_kept = row_mapped[b, keep_index[b]]
            col_kept = sampled_col[b, keep_index[b]]

            unique, idx, counts = torch.unique(col_kept, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat([torch.tensor([0], device=device), cum_sum[:-1]])
            coorespond_row = row_kept[ind_sorted[cum_sum]]
            unique_col = col_kept[ind_sorted[cum_sum]]

            selected_indices = coorespond_row * num_keypoints + unique_col
            positives = torch.exp(torch.index_select(matching_score_original[b].flatten(), 0, selected_indices))
            denominator = torch.sum(torch.exp(matching_score_original[b, :, unique_col]), dim=0)

            infoNCE_loss_g2s[b] = -torch.mean(torch.log(positives / denominator))

    return (infoNCE_loss_s2g + infoNCE_loss_g2s) / 2
