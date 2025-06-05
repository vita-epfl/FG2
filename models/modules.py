import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import desc_l2norm
from torch.hub import load_state_dict_from_url
from DINO_modules.dinov2 import vit_large
from att_layers.transformer import Transformer_self_att
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

import matplotlib.pyplot as plt

import configparser
import ast
config = configparser.ConfigParser()
config.read("./config.ini")
import ast

ground_image_size = ast.literal_eval(config.get("VIGOR", "ground_image_size"))
kitti_grd_size = ast.literal_eval(config.get("KITTI", "ground_image_size"))


eps = config.getfloat("Constants", "epsilon")

seed = config.getint("RandomSeed", "seed")
import random
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(mode=True, warn_only=True)

class DinoExtractor(nn.Module):
    """
    DINOv2 Feature Extractor using a ViT-L/14 backbone.
    """

    def __init__(self, dinov2_weights=None):
        super().__init__()

        # Define DINOv2 extractor parameters
        self.dino_channels = 1024
        self.dino_downfactor = 14
        self.amp_dtype = torch.float16  # Define float precision

        if dinov2_weights is None:
            dinov2_weights = load_state_dict_from_url(
                "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
                map_location="cpu"
            )

        vit_kwargs = dict(
            img_size=518,
            patch_size=14,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
        )

        self.dinov2_vitl14 = vit_large(**vit_kwargs)
        self.dinov2_vitl14.load_state_dict(dinov2_weights)
        self.dinov2_vitl14.requires_grad_(False)
        self.dinov2_vitl14.eval()
        self.dinov2_vitl14.to(self.amp_dtype)

    def forward(self, x):
        B, C, H, W = x.shape
        # Ensure spatial dimensions are divisible by dino_downfactor
        x = x[:, :, : self.dino_downfactor * (H // self.dino_downfactor),
                 : self.dino_downfactor * (W // self.dino_downfactor)]

        with torch.no_grad():
            features = self.dinov2_vitl14.forward_features(x.to(self.amp_dtype))
            features = features["x_norm_patchtokens"].permute(0, 2, 1).reshape(
                B, self.dino_channels, H // self.dino_downfactor, W // self.dino_downfactor
            ).float()

        return features


class BasicBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, padding_mode='zeros'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm2d(planes) if bn else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm2d(planes) if bn else nn.Identity()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x, relu=True):
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += shortcut
        if relu:
            out = F.relu(out)
        return out


class DeepResBlock_desc(torch.nn.Module):
    def __init__(self, bn, last_dim, in_channels, block_dims, add_posEnc, norm_desc, padding_mode = 'zeros'):
        super().__init__()

        self.norm_desc = norm_desc

        self.resblock1 = BasicBlock(in_channels, block_dims[0], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock2 = BasicBlock(block_dims[0], block_dims[1], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock3 = BasicBlock(block_dims[1], block_dims[2], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock4 = BasicBlock(block_dims[2], last_dim, stride=1, bn=bn, padding_mode=padding_mode)

        self.att_layer = Transformer_self_att(d_model=128, num_layers=3, add_posEnc=add_posEnc)


    def forward(self, feature_volume):

        x = self.resblock1(feature_volume)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.att_layer(x)
        x = self.resblock4(x, relu=False)

        if self.norm_desc:
            x = desc_l2norm(x)

        return x

class self_attention(nn.Module):
    def __init__(self, device, num_horizontal, embed_dim):
        super(self_attention, self).__init__()
        self.device = device

        self.embed_dim = embed_dim
        self.num_horizontal = num_horizontal
        
        grd_row_self_ = np.linspace(0, 1, num_horizontal) 
        grd_col_self_ = np.linspace(0, 1, num_horizontal) 
        grd_row_self, grd_col_self = np.meshgrid(grd_row_self_, grd_col_self_, indexing='ij') 
        
        self.grd_reference_points = torch.stack((torch.tensor(grd_col_self), torch.tensor(grd_row_self)), -1).view(-1,2).unsqueeze(1).to(torch.float).to(device)        
        
        self.grd_spatial_shape = torch.tensor(([[num_horizontal, num_horizontal]])).to(device).long()
        self.level_start_index = torch.tensor([0]).to(device)

        self.grd_attention_self = MultiScaleDeformableAttention(embed_dims=embed_dim, num_heads=8, num_levels=1, num_points=4, 
                                                           batch_first=True)

    def forward(self, query):
        bs = query.size()[0]
        residual = query
        
        value = query
        
        grd_reference_points = self.grd_reference_points.unsqueeze(0).repeat(bs, 1, 1, 1)
        grd_bev = self.grd_attention_self(query=query, value=value, reference_points=grd_reference_points, 
                                            spatial_shapes=self.grd_spatial_shape, level_start_index=self.level_start_index)
        
        return grd_bev + residual


class cross_attention(nn.Module):
    def __init__(self, device, num_horizontal, num_vertical, embed_dim, grid_size_h, grid_size_v):
        super(cross_attention, self).__init__()
        self.device = device

        self.embed_dim = embed_dim
        self.num_horizontal = num_horizontal
        self.num_vertical = num_vertical

        # define a 3D grid of points
        x_ = np.linspace(-grid_size_h/2, grid_size_h/2, num_horizontal)
        y_ = np.linspace(-grid_size_h/2, grid_size_h/2, num_horizontal) 
        z_ = np.linspace(-grid_size_v/2, grid_size_v/2, num_vertical) 
        
        grd_x, grd_y, grd_z = np.meshgrid(x_, y_, z_, indexing='ij') # 3D voxel grid with given number (horizontal and vertical) cells
        

        # grd image reference sampling locations 
        phi = np.sign(grd_y+eps) * np.arccos(grd_x / (np.sqrt(grd_x**2 + grd_y**2)+eps)) 
        theta = np.arccos(grd_z / (np.sqrt(grd_x**2 + grd_y**2 + grd_z**2)+eps))
        
        phi = 2*np.pi-phi # to align the orientation direction in ground panorama
        phi[phi>2*np.pi] -= 2*np.pi

        self.grd_col_cross = torch.tensor(phi / (2*np.pi)).to(device)    # [num_horizontal, num_horizontal, num_vertical] # col and row in the panorama
        self.grd_row_cross = torch.tensor(theta / np.pi).to(device)  # [num_horizontal, num_horizontal, num_vertical] 
        
        self.grd_spatial_shape_cross = torch.tensor(([[ground_image_size[0]/14, ground_image_size[1]/14]])).to(device).long()
        self.level_start_index = torch.tensor([0]).to(device)

        self.grd_attention_cross = MultiScaleDeformableAttention(embed_dims=embed_dim, num_heads=8, num_levels=1, num_points=4, 
                                                           batch_first=True)
        self.projector = torch.nn.Linear(embed_dim, 1)


    def forward(self, query, value):
        bs = query.size()[0]
        residual = query

        grd_bev_list = []
        for i in range(self.num_vertical):
            grd_reference_points = torch.stack((self.grd_col_cross[:,:,i], self.grd_row_cross[:,:,i]), -1).view(-1,2).unsqueeze(1).to(torch.float).unsqueeze(0).repeat(bs, 1, 1, 1)

            grd_bev = self.grd_attention_cross(query=query, value=value, reference_points=grd_reference_points, 
                                                spatial_shapes=self.grd_spatial_shape_cross, level_start_index=self.level_start_index)
            
            grd_bev_list.append(grd_bev.view(bs, self.num_horizontal, self.num_horizontal, self.embed_dim))
            
        grd_3d = torch.stack(grd_bev_list, dim=-1).permute(0,1,2,4,3)
        
        weights = torch.nn.functional.softmax(self.projector(grd_3d), dim=3)
        max_height_index = torch.argmax(weights, dim=3)
              
        grd_bev = (weights * grd_3d).sum(3)

        return grd_bev.view(bs, -1, self.embed_dim) + residual, max_height_index

class self_attention_kitti(nn.Module):
    def __init__(self, device, num_horizontal, embed_dim=128):
        super(self_attention_kitti, self).__init__()
        self.device = device

        self.embed_dim = embed_dim
        self.num_horizontal = num_horizontal
        
        grd_row_self_ = np.linspace(0, 1, int(np.floor(self.num_horizontal/2))+1) 
        grd_col_self_ = np.linspace(0, 1, num_horizontal) 
        grd_row_self, grd_col_self = np.meshgrid(grd_row_self_, grd_col_self_, indexing='ij') 
        
        self.grd_reference_points = torch.stack((torch.tensor(grd_col_self), torch.tensor(grd_row_self)), -1).view(-1,2).unsqueeze(1).to(torch.float).to(device)        
        
        self.grd_spatial_shape = torch.tensor(([[int(np.floor(self.num_horizontal/2))+1, num_horizontal]])).to(device).long()
        self.level_start_index = torch.tensor([0]).to(device)

        self.grd_attention_self = MultiScaleDeformableAttention(embed_dims=embed_dim, num_heads=8, num_levels=1, num_points=4, 
                                                           batch_first=True)

    def forward(self, query):
        bs = query.size()[0]
        residual = query
        
        value = query
        
        grd_reference_points = self.grd_reference_points.unsqueeze(0).repeat(bs, 1, 1, 1)
        grd_bev = self.grd_attention_self(query=query, value=value, reference_points=grd_reference_points, 
                                            spatial_shapes=self.grd_spatial_shape, level_start_index=self.level_start_index)
        
        return grd_bev + residual

        
class cross_attention_kitti(nn.Module):
    def __init__(self, device, num_horizontal, num_vertical, embed_dim, grid_size_h, grid_size_v):
        super(cross_attention_kitti, self).__init__()
        self.device = device

        self.embed_dim = embed_dim
        self.num_horizontal = num_horizontal
        self.num_vertical = num_vertical

        # define a 3D grid of point cloud
        x_ = np.linspace(-grid_size_h/2, 0, int(np.floor(self.num_horizontal/2))+1)
        y_ = np.linspace(-grid_size_h/2, grid_size_h/2, num_horizontal) 
        z_ = np.linspace(-grid_size_v/2, grid_size_v/2, num_vertical) 
        
        grd_x, grd_y, grd_z = np.meshgrid(x_, y_, z_, indexing='ij') # 3D voxel grid with given number (horizontal and vertical) cells
        self.grid_3d = torch.stack([torch.tensor(grd_y), torch.tensor(-grd_z), torch.tensor(-grd_x)], dim=-1).reshape(-1, 3).to(torch.float).to(device) # to align camera coordinate system, z pointing outwards, y pointing down, x pointing to the right
        self.grd_spatial_shape_cross = torch.tensor(([[26, 88]])).to(device).long()
        self.level_start_index = torch.tensor([0]).to(device)

        self.grd_attention_cross = MultiScaleDeformableAttention(embed_dims=embed_dim, num_heads=8, num_levels=1, num_points=4, 
                                                           batch_first=True)
        self.projector = torch.nn.Linear(embed_dim, 1)

    def forward(self, query, value, camera_k):
        bs = query.size()[0]
        residual = query

        grid_3d = self.grid_3d.T.unsqueeze(0).repeat(bs, 1, 1)
        projected_points = camera_k @ grid_3d
        u = (projected_points[:, 0, :] / projected_points[:, 2, :] / kitti_grd_size[1]).view(bs, int(np.floor(self.num_horizontal/2))+1, self.num_horizontal, self.num_vertical)
        v = (projected_points[:, 1, :] / projected_points[:, 2, :] / kitti_grd_size[0]).view(bs, int(np.floor(self.num_horizontal/2))+1, self.num_horizontal, self.num_vertical)

        keep_index = torch.logical_and(torch.logical_and(u >= 0, u <= 1), torch.logical_and(v >= 0, v <= 1))
        keep_index = keep_index.sum(-1).to(torch.bool)
        
        grd_bev_list = []
        for i in range(self.num_vertical):
            grd_reference_points = torch.stack((u[:,:,:,i], v[:,:,:,i]), -1).view(bs,-1,2).unsqueeze(2).to(torch.float)

            grd_bev = self.grd_attention_cross(query=query, value=value, reference_points=grd_reference_points, 
                                                spatial_shapes=self.grd_spatial_shape_cross, level_start_index=self.level_start_index)
            
            grd_bev_list.append(grd_bev.view(bs, int(np.floor(self.num_horizontal/2))+1, self.num_horizontal, self.embed_dim))
            
        grd_3d = torch.stack(grd_bev_list, dim=-1).permute(0,1,2,4,3)
        
        weights = torch.nn.functional.softmax(self.projector(grd_3d), dim=3)
        max_height_index = torch.argmax(weights, dim=3)
              
        grd_bev = (weights * grd_3d).sum(3)


        return grd_bev.view(bs, -1, self.embed_dim) + residual, max_height_index, keep_index