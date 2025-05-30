import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import self_attention, cross_attention, DeepResBlock_desc
from mmcv.cnn.bricks.transformer import FFN

import random
import numpy as np

import configparser
config = configparser.ConfigParser()
config.read("./config.ini")
seed = config.getint("RandomSeed", "seed")
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(mode=True, warn_only=True)

class CVM(nn.Module):
    def __init__(self, device, grd_bev_res, grd_height_res, sat_bev_res, grid_size_h, grid_size_v,
                 temperature=0.1, embed_dim=1024, desc_dim=128):
        super(CVM, self).__init__()
        self.device = device
        self.temperature = temperature
        self.embed_dim = embed_dim

        self.grd_bev_res = grd_bev_res
        self.sat_bev_res = sat_bev_res
        self.num_col = sat_bev_res


        bn = True
        block_dims = [512, 256, 128, 64]
        add_posEnc = True
        norm_desc = True

        
        self.dustbin_score = nn.Parameter(torch.tensor(1.))

        ### grd
        self.grd_grid_queries = nn.Parameter(data=torch.rand(grd_bev_res, grd_bev_res, embed_dim), requires_grad=True).to(device)

        self.grd_attention_self1 = self_attention(device, grd_bev_res, embed_dim)
        self.grd_attention_cross1 = cross_attention(device, grd_bev_res, grd_height_res, embed_dim, grid_size_h, grid_size_v)

        self.grd_attention_self2 = self_attention(device, grd_bev_res, embed_dim)
        self.grd_attention_cross2 = cross_attention(device, grd_bev_res, grd_height_res, embed_dim, grid_size_h, grid_size_v)

        self.grd_attention_self3 = self_attention(device, grd_bev_res, embed_dim)
        self.grd_attention_cross3 = cross_attention(device, grd_bev_res, grd_height_res, embed_dim, grid_size_h, grid_size_v)

        self.grd_attention_self4 = self_attention(device, grd_bev_res, embed_dim)
        self.grd_attention_cross4 = cross_attention(device, grd_bev_res, grd_height_res, embed_dim, grid_size_h, grid_size_v)

        self.grd_attention_self5 = self_attention(device, grd_bev_res, embed_dim)
        self.grd_attention_cross5 = cross_attention(device, grd_bev_res, grd_height_res, embed_dim, grid_size_h, grid_size_v)

        self.grd_attention_self6 = self_attention(device, grd_bev_res, embed_dim)
        self.grd_attention_cross6 = cross_attention(device, grd_bev_res, grd_height_res, embed_dim, grid_size_h, grid_size_v)

        self.grd_ffn1 = FFN(embed_dim)
        self.grd_ffn2 = FFN(embed_dim)
        self.grd_ffn3 = FFN(embed_dim)
        self.grd_ffn4 = FFN(embed_dim)
        self.grd_ffn5 = FFN(embed_dim)
        self.grd_ffn6 = FFN(embed_dim)

        self.grd_layer_norm11 = nn.LayerNorm(embed_dim)
        self.grd_layer_norm12 = nn.LayerNorm(embed_dim)
        self.grd_layer_norm13 = nn.LayerNorm(embed_dim)

        self.grd_layer_norm21 = nn.LayerNorm(embed_dim)
        self.grd_layer_norm22 = nn.LayerNorm(embed_dim)
        self.grd_layer_norm23 = nn.LayerNorm(embed_dim)

        self.grd_layer_norm31 = nn.LayerNorm(embed_dim)
        self.grd_layer_norm32 = nn.LayerNorm(embed_dim)
        self.grd_layer_norm33 = nn.LayerNorm(embed_dim)

        self.grd_layer_norm41 = nn.LayerNorm(embed_dim)
        self.grd_layer_norm42 = nn.LayerNorm(embed_dim)
        self.grd_layer_norm43 = nn.LayerNorm(embed_dim)

        self.grd_layer_norm51 = nn.LayerNorm(embed_dim)
        self.grd_layer_norm52 = nn.LayerNorm(embed_dim)
        self.grd_layer_norm53 = nn.LayerNorm(embed_dim)

        self.grd_layer_norm61 = nn.LayerNorm(embed_dim)
        self.grd_layer_norm62 = nn.LayerNorm(embed_dim)
        self.grd_layer_norm63 = nn.LayerNorm(embed_dim)

        self.grd_projector =  DeepResBlock_desc(bn, last_dim=desc_dim, in_channels=embed_dim, block_dims=block_dims, add_posEnc=add_posEnc, norm_desc=norm_desc)
        
        ### sat
        self.sat_projector =  DeepResBlock_desc(bn, last_dim=desc_dim, in_channels=embed_dim, block_dims=block_dims, add_posEnc=add_posEnc, norm_desc=norm_desc)
    
    def forward(self, grd_feature, sat_feature):

        bs = grd_feature.size()[0]

        grd_query = self.grd_grid_queries.view(-1, self.embed_dim).unsqueeze(0).repeat(bs, 1, 1)

        grd_value = grd_feature.permute(0,2,3,1).view(bs, -1, self.embed_dim).contiguous()

        # iter 1
        grd_query = self.grd_attention_self1(grd_query)        
        grd_query = self.grd_layer_norm11(grd_query)
        
        grd_query, _ = self.grd_attention_cross1(grd_query, grd_value)
        grd_query = self.grd_layer_norm12(grd_query)

        grd_query = self.grd_ffn1(grd_query)
        grd_query = self.grd_layer_norm13(grd_query)

        # iter 2
        grd_query = self.grd_attention_self2(grd_query)
        grd_query = self.grd_layer_norm21(grd_query)
        
        grd_query, _ = self.grd_attention_cross2(grd_query, grd_value)
        grd_query = self.grd_layer_norm22(grd_query)

        grd_query = self.grd_ffn2(grd_query)
        grd_query = self.grd_layer_norm23(grd_query)

        # iter 3
        grd_query = self.grd_attention_self3(grd_query)
        grd_query = self.grd_layer_norm31(grd_query)
        
        grd_query, _ = self.grd_attention_cross3(grd_query, grd_value)
        grd_query = self.grd_layer_norm32(grd_query)

        grd_query = self.grd_ffn3(grd_query)
        grd_query = self.grd_layer_norm33(grd_query)
        
        # iter 4
        grd_query = self.grd_attention_self4(grd_query)
        grd_query = self.grd_layer_norm41(grd_query)
        
        grd_query, _ = self.grd_attention_cross4(grd_query, grd_value)
        grd_query = self.grd_layer_norm42(grd_query)

        grd_query = self.grd_ffn1(grd_query)
        grd_query = self.grd_layer_norm43(grd_query)

        # iter 5
        grd_query = self.grd_attention_self5(grd_query)
        grd_query = self.grd_layer_norm51(grd_query)
        
        grd_query, _ = self.grd_attention_cross5(grd_query, grd_value)
        grd_query = self.grd_layer_norm52(grd_query)

        grd_query = self.grd_ffn5(grd_query)
        grd_query = self.grd_layer_norm53(grd_query)

        # iter 6
        grd_query = self.grd_attention_self6(grd_query)
        grd_query = self.grd_layer_norm61(grd_query)
        
        grd_query, height_index = self.grd_attention_cross6(grd_query, grd_value)
        grd_query = self.grd_layer_norm62(grd_query)

        grd_query = self.grd_ffn6(grd_query)
        grd_query = self.grd_layer_norm63(grd_query)
        
        grd_query = grd_query.permute(0,2,1).view(bs, self.embed_dim, self.grd_bev_res, self.grd_bev_res)
        grd_desc = self.grd_projector(grd_query).flatten(2)
        
        sat_feature_bev = nn.functional.interpolate(sat_feature, (self.sat_bev_res, self.sat_bev_res), mode='bilinear')        
        sat_desc = self.sat_projector(sat_feature_bev).flatten(2)

        
        matching_score_original = torch.matmul(sat_desc.transpose(1, 2).contiguous(), grd_desc) / self.temperature
        
        b, m, n = matching_score_original.shape

        bins0 = self.dustbin_score.expand(b, m, 1)
        bins1 = self.dustbin_score.expand(b, 1, n)
        alpha = self.dustbin_score.expand(b, 1, 1)

        couplings = torch.cat([torch.cat([matching_score_original, bins0], -1),
                               torch.cat([bins1, alpha], -1)], 1)

        couplings = F.softmax(couplings, 1) * F.softmax(couplings, 2)
        matching_score = couplings[:, :-1, :-1]

        
        return matching_score, matching_score_original, height_index

