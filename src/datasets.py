from src.settings import NUSCENES, RV_WIDTH, RV_HEIGHT
from src.preprocess import pcl_to_rangeview

import numpy as np
import torch

from torch.utils.data import Dataset
from os.path import join
from pyquaternion import Quaternion


class NuscenesDataset(Dataset):

    def __init__(self, data_root, n: tuple=None):
        """
        Args:
            root_dir (string): root NuScenes directory
            transform (callable, optional): Optional transform to be applied on a sample.
            
            n - tuple with left and right index boundaries, in case we don't want to use all data
        """
        assert len(n) == 2
        self.data_root = data_root
        self.nuscenes = NUSCENES

        if n:
            self.samples = self.nuscenes.sample[n[0]:n[1]]
        else:
            self.samples = self.nuscenes.sample
            
        # point_clouds_features will be of shape (N, M, 5), 
        # where N - number of samples, M - number of points in according sample's pointcloud
        self.point_clouds_features = []
        self.point_clouds_labels = []
        
        self.__set_point_clouds()
            
    def __set_point_clouds(self):

        for sample in self.samples:
            
            sample_data_token = sample['data']['LIDAR_TOP']

            my_sample_lidar_data = self.nuscenes.get('sample_data', sample_data_token)

            lidarseg_labels_filename = join(self.data_root,
                                            self.nuscenes.get('lidarseg', sample_data_token)['filename'])

            # loading directly from files to perceive the ring_index information
            points_raw = np.fromfile(self.data_root + my_sample_lidar_data["filename"], dtype=np.float32).reshape((-1, 5))
            point_labels = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)
            
            self.point_clouds_features.append(points_raw)
            self.point_clouds_labels.append(point_labels)
            
        self.point_clouds_features = np.array(self.point_clouds_features)
        self.point_clouds_labels = np.array(self.point_clouds_labels)

    def __len__(self):
        return len(self.samples)

    def get_front_bb(self, sample: dict):
        """
        Function computes and returns
        An array of points points of a bounding box in sensor coordinates.
        Each point is a (x, y) coordinate, each BB is 4 points

        :param sample: nuscenes sample dictionary
        :return: np.array of shape (N, 8, 3)
        """
        my_sample_lidar_data = self.nuscenes.get('sample_data', sample['data']['LIDAR_TOP'])
        sample_annotation = self.nuscenes.get_boxes(my_sample_lidar_data['token'])

        ego_record = self.nuscenes.get('ego_pose', my_sample_lidar_data['ego_pose_token'])
        cs_record = self.nuscenes.get('calibrated_sensor', my_sample_lidar_data['calibrated_sensor_token'])

        # first step: transform from absolute to ego
        ego_translation = -np.array(ego_record['translation'])

        # second step: transform from ego to sensor
        cs_translation = -np.array(cs_record['translation'])

        corners = []
        for box in sample_annotation:
            box.translate(ego_translation)
            box.rotate(Quaternion(ego_record['rotation']).inverse)

            box.translate(cs_translation)
            box.rotate(Quaternion(cs_record['rotation']).inverse)

            # at this point bounding boxes are in sensor coordinate system
            # now we want to exclude such BB that do not have their center
            # lying in the front 90 degrees

            if box.center[1] <= 0:
                continue

            box.center_azimuth = np.degrees(np.arctan(box.center[0] / box.center[1]))

            # Transform front azimuth to be in range from 0 to 180
            box.center_azimuth = 90 - box.center_azimuth
            if not (45 < box.center_azimuth < 135):
                continue

            corners.append(box.bottom_corners())
        
        if len(corners) > 0:
            return np.transpose(np.array(corners), (0, 2, 1))
        else:
            return np.zeros((1, 4, 3))
    
    def points_in_box(self, coordinates, bounding_box_corners):
        """
            bounding_box_corners: bbc of a single bb
            return a mask of whether points that are in the box
        """
        coords_x = coordinates[0]
        coords_y = coordinates[1]
        
        min_bb_x = bounding_box_corners[:, 0].min()
        max_bb_x = bounding_box_corners[:, 0].max()
        min_bb_y = bounding_box_corners[:, 1].min()
        max_bb_y = bounding_box_corners[:, 1].max()

        c1 = min_bb_x <= coords_x  # left_top/left_bottom.x <= coordinate.x
        c2 = max_bb_x >= coords_x  # right_bottom/right_top.x >= coordinate.x
        c3 = min_bb_y <= coords_y  # left/right_bottom.y <= coordinate.y
        c4 = max_bb_y >= coords_y  # right_top/left_top.y >= coordinate.y

        c = np.logical_and(np.logical_and(c1, c2),
                           np.logical_and(c3, c4))
        return c
    
    def get_bb_targets(self, idx, bounding_box_corners):
        coordinates = self.point_clouds_features[idx][:2] 
    
        for bb_c in bounding_box_corners:

            if self.points_in_box(coordinates, bb_c).any():
                return np.array(bb_c[:, :2])
            
            else:
                return np.zeros_like(bb_c[:, :2])
    
    def __getitem__(self, idx, compute_boxes=True):
        """
        compute_box will be set to False in child classes, so target boxes are computed after rotations
        
        """
        
        sample_idx = self.samples[idx]
        pcl_features = self.point_clouds_features[idx]
        pcl_labels = self.point_clouds_labels[idx]
        
        if compute_boxes:
            front_bbs = self.get_front_bb(sample_idx)
            target_bounding_boxes = self.get_bb_targets(idx, front_bbs)
    
            return pcl_features, pcl_labels, target_bounding_boxes
        
        else:
            return pcl_features, pcl_labels, []
        

class NuscenesRangeViewDataset(NuscenesDataset):

    def __init__(self, data_root, n=None):
        super().__init__(data_root, n)
        
            
    def __len__(self):
        return len(self.samples)
    
    def get_bb_targets(self, range_view_coordinates, bounding_box_corners):
        bbc_target = np.zeros((4, 2, RV_WIDTH, RV_HEIGHT))
                
        for bbc in bounding_box_corners:
            point_mask = self.points_in_box(range_view_coordinates, bbc)
            bbc_target[:, :, point_mask] = np.expand_dims(bbc, 2)
            
        return bbc_target.reshape((8, RV_WIDTH, RV_HEIGHT))
        
        
    def __getitem__(self, idx):
        pcl_features, pcl_targets, _ = super().__getitem__(idx, compute_boxes=False)
        
        rotate_prob = np.random.uniform()
        
#         if rotate_prob > 0.5:
#             rotation_angle_y = np.random.uniform(10, 90)

#             rotation = RandomRotation((0, rotation_angle_y))
#             print("0", pcl_features[:, :2].shape)
#             pcl_features[:, :2] = pcl_features[:, :2] @ rotation_matrix(torch.Tensor([rotation_angle_y])).cpu().numpy()
#             print("1", pcl_features[:, :2].shape)
            
        range_view, targets = pcl_to_rangeview(pcl_features, pcl_targets)
        
        range_view = range_view.transpose(2, 1, 0)
        targets = targets.transpose(2, 1, 0)
        
        front_bbs = self.get_front_bb(self.samples[idx])[:, :, :2]  # N x 4 x 2 since we only need xy
        target_bounding_boxes = self.get_bb_targets(range_view[:2], front_bbs)
    
        return torch.Tensor(range_view.copy()), torch.Tensor(targets.copy()), torch.Tensor(target_bounding_boxes.copy())