# [CVPR'25] FG²: Fine-Grained Cross-View Localization by Fine-Grained Feature Matching
[[`Arxiv`](https://arxiv.org/abs/2503.18725)][[`BibTeX`](#citation)]

![](figures/method.png)




## Abstract
We propose a novel fine-grained cross-view localization method that estimates the 3 Degrees of Freedom pose of a ground-level image in an aerial image of the surroundings by matching fine-grained features between the two images. The pose is estimated by aligning a point plane generated from the ground image with a point plane sampled from the aerial image. To generate the ground points, we first map ground image features to a 3D point cloud. Our method then learns to select features along the height dimension to pool the 3D points to a Bird’s-Eye-View (BEV) plane. This selection enables us to trace which feature in the ground image contributes to the BEV representation. Next, we sample a set of sparse matches from computed point correspondences between the two point planes and compute their relative pose using Procrustes alignment. Compared to the previous state-of-the-art, our method reduces the mean localization error by 28% on the VIGOR cross-area test set. Qualitative results show that our method learns semantically consistent matches across ground and aerial view through weakly supervised learning from the camera pose.

## Checkpoints
The trained models are available at https://drive.google.com/drive/folders/1WUViQcX9m0PE9FePbWklisBS88sHOyMK?usp=sharing


## 📋 To-Do List

- [x] Initial project setup and repo structure
- [x] Add license and basic README documentation
- [ ] Add installation instructions
- [x] Add model training scripts
- [ ] Add model testing scripts
- [x] Include pretrained models
- [ ] Add code for visualization

## Citation
```
@article{xia2025fg,
  title={FG $\^{} 2$: Fine-Grained Cross-View Localization by Fine-Grained Feature Matching},
  author={Xia, Zimin and Alahi, Alexandre},
  journal={arXiv preprint arXiv:2503.18725},
  year={2025}
}
```
