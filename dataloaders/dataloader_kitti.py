import os
from torch.utils.data import Dataset
import PIL.Image
import torch
import numpy as np
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms.functional as TF
from torchvision import transforms

# Load configuration
import configparser
import ast
config = configparser.ConfigParser()
config.read("./config.ini")

# Set deterministic behavior for reproducibility
seed = config.getint("RandomSeed", "seed")
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(mode=True, warn_only=True)

ground_image_size = ast.literal_eval(config.get("KITTI", "ground_image_size"))
satellite_image_size = ast.literal_eval(config.get("KITTI", "satellite_image_size"))

GrdImg_H = ground_image_size[0]
GrdImg_W = ground_image_size[1]
SatMap_process_sidelength = satellite_image_size[0]

Default_lat = 49.015
Satmap_zoom = 18

GrdOriImg_H = 375
GrdOriImg_W = 1242
SatMap_original_sidelength = 512 

satmap_dir = 'satmap'
grdimage_dir = 'raw_data'
oxts_dir = 'oxts/data'  
left_color_camera_dir = 'image_02/data'
CameraGPS_shift_left = [1.08, 0.26]

satmap_transform = transforms.Compose([
        transforms.Resize(size=satellite_image_size),
        transforms.ToTensor()
    ])
    
grdimage_transform = transforms.Compose([
        transforms.Resize(size=ground_image_size),
        transforms.ToTensor()
    ])


dataset_root = config["KITTI"]["dataset_root"]
train_file = config["KITTI"]["train_file"]
test1_file = config["KITTI"]["test1_file"]
test2_file = config["KITTI"]["test2_file"]

shift_range_lat = config.getfloat("KITTI", "shift_range_lat")
shift_range_lon = config.getfloat("KITTI", "shift_range_lon")
rotation_range = config.getfloat("KITTI", "rotation_range")


def get_meter_per_pixel(lat=Default_lat, zoom=Satmap_zoom, scale=SatMap_process_sidelength/SatMap_original_sidelength):
    meter_per_pixel = 156543.03392 * np.cos(lat * np.pi/180.) / (2**zoom)	
    meter_per_pixel /= 2 # because use scale 2 to get satmap 
    meter_per_pixel /= scale
    return meter_per_pixel

class SatGrdDataset(Dataset):
    def __init__(self, root, file,
                 transform=None, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
        self.root = root

        self.meter_per_pixel = get_meter_per_pixel(scale=1)
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel  # shift range is in terms of pixels
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel  # shift range is in terms of pixels

        self.rotation_range = rotation_range  # in terms of degree

        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~
        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]

        self.pro_grdimage_dir = 'raw_data'

        self.satmap_dir = satmap_dir

        with open(file, 'r') as f:
            file_name = f.readlines()

        self.file_name = [file[:-1] for file in file_name]

    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    
    def __getitem__(self, idx):
        # read cemera k matrix from camera calibration files, day_dir is first 10 chat of file name

        file_name = self.file_name[idx]
        day_dir = file_name[:10]
        drive_dir = file_name[:38]#2011_09_26/2011_09_26_drive_0002_sync/
        image_no = file_name[38:]

        # =================== read camera intrinsice for left and right cameras ====================
        calib_file_name = os.path.join(self.root, grdimage_dir, day_dir, 'calib_cam_to_cam.txt')
        with open(calib_file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # left color camera k matrix
                if 'P_rect_02' in line:
                    # get 3*3 matrix from P_rect_**:
                    items = line.split(':')
                    valus = items[1].strip().split(' ')
                    fx = float(valus[0]) * GrdImg_W / GrdOriImg_W
                    cx = float(valus[2]) * GrdImg_W / GrdOriImg_W
                    fy = float(valus[5]) * GrdImg_H / GrdOriImg_H
                    cy = float(valus[6]) * GrdImg_H / GrdOriImg_H
                    left_camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                    left_camera_k = torch.from_numpy(np.asarray(left_camera_k, dtype=np.float32))
                    break
        
        # =================== read satellite map ===================================
        SatMap_name = os.path.join(self.root, self.satmap_dir, file_name)
        with PIL.Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])
        image_no = file_name[38:]

        # oxt: such as 0000000000.txt
        oxts_file_name = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                      image_no.lower().replace('.png', '.txt'))
        with open(oxts_file_name, 'r') as f:
            content = f.readline().split(' ')
            # get heading
            lat = float(content[0])
            lon = float(content[1])
            heading = float(content[5])

            left_img_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, left_color_camera_dir,
                                         image_no.lower())
            
            with PIL.Image.open(left_img_name, 'r') as GrdImg:
                grd_img_left = GrdImg.convert('RGB')
                if self.grdimage_transform is not None:
                    grd_img_left = self.grdimage_transform(grd_img_left)

            grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)

                    
        sat_rot = sat_map.rotate((90-heading) / np.pi * 180) # make the up direction the vehicle heading
        sat_align_cam = sat_rot.transform(sat_rot.size, PIL.Image.AFFINE,
                                          (1, 0, CameraGPS_shift_left[0] / self.meter_per_pixel,
                                           0, 1, CameraGPS_shift_left[1] / self.meter_per_pixel),
                                          resample=PIL.Image.BILINEAR) 
        # randomly generate shift
        gt_shift_x = np.random.uniform(-1, 1)  # --> right as positive
        gt_shift_y = np.random.uniform(-1, 1)  # --> up as positive
        
        sat_rand_shift = \
            sat_align_cam.transform(
                sat_align_cam.size, PIL.Image.AFFINE,
                (1, 0, gt_shift_x * self.shift_range_pixels_lon,
                 0, 1, -gt_shift_y * self.shift_range_pixels_lat),
                resample=PIL.Image.BILINEAR)
        
        # randomly generate roation
        random_ori = np.random.uniform(-1, 1) * self.rotation_range # 0 means the grd is heading up in aerial image, counter-clockwise increasing
        
        sat_rand_shift_rand_rot = sat_rand_shift.rotate(random_ori)
        
        sat_map =TF.center_crop(sat_rand_shift_rand_rot, SatMap_process_sidelength)
        
        # transform
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)

        # location gt
        x_offset = int(gt_shift_x*self.shift_range_pixels_lon*np.cos(random_ori/180*np.pi) - gt_shift_y*self.shift_range_pixels_lat*np.sin(random_ori/180*np.pi)) # horizontal direction
        y_offset = int(-gt_shift_y*self.shift_range_pixels_lat*np.cos(random_ori/180*np.pi) - gt_shift_x*self.shift_range_pixels_lon*np.sin(random_ori/180*np.pi)) # vertical direction
        
        x_offset = x_offset/SatMap_original_sidelength*SatMap_process_sidelength
        y_offset = y_offset/SatMap_original_sidelength*SatMap_process_sidelength
        
        gt_loc = torch.tensor([[y_offset, x_offset]]) 

        # orientation gt
        yaw = - random_ori * (np.pi/180) # rotation from sat to grd 
        
        r = np.zeros((2,2), dtype=np.float32)
        r[0,0] = np.cos(yaw)
        r[0,1] = -np.sin(yaw)
        r[1,0] = np.sin(yaw) 
        r[1,1] = np.cos(yaw)
        r = torch.tensor(r)
        
        return sat_map, grd_left_imgs[0], left_camera_k, gt_loc, r
               
class SatGrdDatasetTest(Dataset):
    def __init__(self, root, file,
                 transform=None, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
        self.root = root

        self.meter_per_pixel = get_meter_per_pixel(scale=1)
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel  # shift range is in terms of meters
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel  # shift range is in terms of meters

        self.rotation_range = rotation_range  # in terms of degree

        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~
        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]

        self.pro_grdimage_dir = 'raw_data'

        self.satmap_dir = satmap_dir

        with open(file, 'r') as f:
            file_name = f.readlines()

        self.file_name = [file[:-1] for file in file_name]
       

    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    
    def __getitem__(self, idx):

        line = self.file_name[idx]
        file_name, gt_shift_x, gt_shift_y, theta = line.split(' ')
        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]

        # =================== read camera intrinsice for left and right cameras ====================
        calib_file_name = os.path.join(self.root, grdimage_dir, day_dir, 'calib_cam_to_cam.txt')
        with open(calib_file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # left color camera k matrix
                if 'P_rect_02' in line:
                    # get 3*3 matrix from P_rect_**:
                    items = line.split(':')
                    valus = items[1].strip().split(' ')
                    fx = float(valus[0]) * GrdImg_W / GrdOriImg_W
                    cx = float(valus[2]) * GrdImg_W / GrdOriImg_W
                    fy = float(valus[5]) * GrdImg_H / GrdOriImg_H
                    cy = float(valus[6]) * GrdImg_H / GrdOriImg_H
                    left_camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                    left_camera_k = torch.from_numpy(np.asarray(left_camera_k, dtype=np.float32))
                    # if not self.stereo:
                    break
        
        
        # =================== read satellite map ===================================
        SatMap_name = os.path.join(self.root, self.satmap_dir, file_name)
        with PIL.Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])
        # image_no = file_name[38:]

        # oxt: such as 0000000000.txt
        oxts_file_name = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                      image_no.lower().replace('.png', '.txt'))
        with open(oxts_file_name, 'r') as f:
            content = f.readline().split(' ')
            # get heading
            lat = float(content[0])
            lon = float(content[1])
            heading = float(content[5])

            left_img_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, left_color_camera_dir,
                                         image_no.lower())
            
            
            with PIL.Image.open(left_img_name, 'r') as GrdImg:
                grd_img_left = GrdImg.convert('RGB')
                if self.grdimage_transform is not None:
                    grd_img_left = self.grdimage_transform(grd_img_left)

            grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)
        
        sat_rot = sat_map.rotate(90-heading / np.pi * 180)
        
        sat_align_cam = sat_rot.transform(sat_rot.size, PIL.Image.AFFINE,
                                          (1, 0, CameraGPS_shift_left[0] / self.meter_per_pixel,
                                           0, 1, CameraGPS_shift_left[1] / self.meter_per_pixel),
                                          resample=PIL.Image.BILINEAR)
        
        # load the shifts 
        gt_shift_x = -float(gt_shift_x)  # --> right as positive
        gt_shift_y = -float(gt_shift_y)  # --> up as positive

        sat_rand_shift = \
            sat_align_cam.transform(
                sat_align_cam.size, PIL.Image.AFFINE,
                (1, 0, gt_shift_x * self.shift_range_pixels_lon,
                 0, 1, -gt_shift_y * self.shift_range_pixels_lat),
                resample=PIL.Image.BILINEAR)
        random_ori = float(theta) * self.rotation_range # degree

        sat_rand_shift_rand_rot = sat_rand_shift.rotate(random_ori)
        
        sat_map =TF.center_crop(sat_rand_shift_rand_rot, SatMap_process_sidelength)

        # transform
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)

        # location gt
        x_offset = (gt_shift_x*self.shift_range_pixels_lon*np.cos(random_ori/180*np.pi) - gt_shift_y*self.shift_range_pixels_lat*np.sin(random_ori/180*np.pi)) # horizontal direction
        y_offset = (-gt_shift_y*self.shift_range_pixels_lat*np.cos(random_ori/180*np.pi) - gt_shift_x*self.shift_range_pixels_lon*np.sin(random_ori/180*np.pi)) # vertical direction
        
        x_offset = x_offset/SatMap_original_sidelength*SatMap_process_sidelength
        y_offset = y_offset/SatMap_original_sidelength*SatMap_process_sidelength

        gt_loc = torch.tensor([[y_offset, x_offset]], dtype=torch.float32)

        # orientation gt

        yaw = -random_ori * (np.pi/180) 
        

        r = np.zeros((2,2), dtype=np.float32)
        r[0,0] = np.cos(yaw)
        r[0,1] = -np.sin(yaw)
        r[1,0] = np.sin(yaw) 
        r[1,1] = np.cos(yaw)
        r = torch.tensor(r)
        
        return sat_map, grd_left_imgs[0], left_camera_k, gt_loc, r

train_set = SatGrdDataset(root=dataset_root, file=train_file,
                              transform=(satmap_transform, grdimage_transform),
                              shift_range_lat=shift_range_lat,
                              shift_range_lon=shift_range_lon,
                              rotation_range=rotation_range)

test1_set = SatGrdDatasetTest(root=dataset_root, file=test1_file,
                              transform=(satmap_transform, grdimage_transform),
                              shift_range_lat=shift_range_lat,
                              shift_range_lon=shift_range_lon,
                              rotation_range=rotation_range)

test2_set = SatGrdDatasetTest(root=dataset_root, file=test2_file,
                              transform=(satmap_transform, grdimage_transform),
                              shift_range_lat=shift_range_lat,
                              shift_range_lon=shift_range_lon,
                              rotation_range=rotation_range)
