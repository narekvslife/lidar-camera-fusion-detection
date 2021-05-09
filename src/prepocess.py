import numpy as np
import pandas as pd
from os.path import join
from nuscenes.utils.data_classes import LidarPointCloud, Quaternion

from .settings import DATASET_PATH, NUSCENES


def transform_pcl_to_global(pcl, sample_data):
    """
        :params: pcl in sensor coordinate system

        :return: pcl in abslute coordinate system
    """

    # First step: transform the pointcloud to the ego vehicle frame
    cs_record = NUSCENES.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    pcl.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pcl.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = NUSCENES.get('ego_pose', sample_data['ego_pose_token'])
    pcl.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pcl.translate(np.array(poserecord['translation']))

    return pcl


def sample_to_rangeview(sample: dict,
                        height: int = 32,
                        width: int = 135) -> np.array:
    """
        Transform sample's Lidar Point Cloud to Range View.

        Range View is an image of size H x W x 5,
        where H - number of lasers, W - number of discretized azimuth bins.
        Each of 5 channels is responsible for a single interesting feature:
        range, height, azimuth, intensity and flag indicating whether there is a point or not

        Lookup into each of feature matrices looks like matrice[laser_number][azimuth_bin]
    """

    sample_data_token = sample['data']['LIDAR_TOP']

    my_sample_lidar_data = NUSCENES.get('sample_data', sample_data_token)

    lidarseg_labels_filename = join(NUSCENES.dataroot,
                                    NUSCENES.get('lidarseg', sample_data_token)['filename'])

    # loading directly from files to perceive the ring_index information
    points_raw = np.fromfile(DATASET_PATH + my_sample_lidar_data["filename"], dtype=np.float32).reshape((-1, 5))
    point_labeles = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)

    pcl = LidarPointCloud.from_file(DATASET_PATH + my_sample_lidar_data["filename"])
    pcl = transform_pcl_to_global(pcl, my_sample_lidar_data)

    points_raw = np.hstack((pcl.points.T[:, :3], points_raw))  # TODO: redo by zeros
    assert (points_raw.shape[1] == 8)

    points_df = pd.DataFrame(points_raw, columns=['abs_x', 'abs_y', 'abs_z',
                                                  'rel_x', 'rel_y', 'rel_z',
                                                  'intensity', 'ring_index'])
    points_df['class'] = point_labeles
    points_df['azimuth'] = np.degrees(np.arctan(points_df['rel_x'] / points_df['rel_y']))

    # Transform front azimuth to be in range from 0 to 180
    mask_front = (points_df.rel_y >= 0)
    points_df = points_df[mask_front]
    points_df['azimuth'] += 90
    points_df['azimuth'] = 180 - points_df.loc[mask_front, 'azimuth']

    # We only care about the front 90 degrees
    front_90_mask = (45 < points_df.azimuth) & (points_df.azimuth < 135)
    points_df = points_df[front_90_mask]
    df_len = len(points_df)
    # Reindex by row number
    points_df.index = np.arange(df_len)

    # distance to the point is one of the interesting features
    points_df['distance'] = np.linalg.norm(np.vstack((points_df.rel_x.values,
                                                      points_df.rel_y.values,
                                                      points_df.rel_z.values)), axis=0)

    # We want to divide front 90 degrees into number of bins equal to width
    bin_size = 90 / width
    points_df['azimuth_bin'] = np.digitize(x=points_df['azimuth'].values - 45,  # right-most value is 45
                                           bins=np.arange(0, 90, bin_size))

    # artificial "point" which will represent values for missing points
    points_df.loc[df_len] = (points_df.abs_x.values[0],
                             points_df.abs_y.values[0],
                             points_df.abs_z.values[0], 0, 0, 0, 0, 0, 0, 0, 0, 1)

    # Finally construct the Range View image
    # First, we construct 4 channels: height, intensity, aziumth, distance
    image = np.zeros((height, width, 5))
    #     try:
    # row numbers of points which have minimal distance in their groups

    idx_min = points_df.groupby(['ring_index', 'azimuth_bin']) \
        .distance.idxmin() \
        .unstack(fill_value=df_len).stack().values
    points_df = points_df.loc[idx_min]

    point_labeles = points_df['class'].values.reshape((height, width))

    points_df.drop(['class'], inplace=True, axis=1)
    points_df.drop(['ring_index', 'azimuth_bin'], inplace=True, axis=1)

    points_features = points_df.values.reshape((height, width, 9))

    # first 3 columns were abs_x, abs_y, abs_z
    absolute_coordinates = np.zeros((height, width, 3))
    absolute_coordinates[:, :, :2] = points_features[:, :, :2]

    # here are the 4 channels: distance, azimuth, reflectance, and height
    image[:, :, :4] = points_features[:, :, 5:]

    # 5th channel is a flag which shows whether there is a point or not.
    # If distance == azimuth == ring_index == 0 => no point
    image[:, :, 4] = (image[:, :, :].sum(axis=2) != 0).astype(int)

    assert image.shape[: 2] == absolute_coordinates.shape[: 2]

    #     except ValueError:
    #         print(sample['token'])

    # need to reflect x and y, so it matches camera view
    return image[::-1, ::-1, :], absolute_coordinates[::-1, ::-1, ::], point_labeles[::-1, ::-1]
