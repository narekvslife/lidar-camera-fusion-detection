import numpy as np
import pandas as pd
from os.path import join

from src.settings import DATASET_PATH, NUSCENES, LABEL_NUMBER, RV_HEIGHT, RV_WIDTH


def pcl_to_rangeview(pcl_features: np.array, pcl_labels: np.array):
    
    points_df = pd.DataFrame(pcl_features, columns=['x', 'y', 'z',
                                                    'intensity', 'ring_index'])
    
    points_df['class'] = pcl_labels
    points_df['azimuth'] = np.degrees(np.arctan(points_df['x'] / points_df['y']))

    # Transform front azimuth to be in range from 0 to 180
    mask_front = (points_df.y >= 0)
    points_df = points_df[mask_front]
    points_df['azimuth'] = 90 - points_df.loc[mask_front, 'azimuth']

    # We only care about the front 90 degrees
    front_90_mask = (45 < points_df.azimuth) & (points_df.azimuth < 135)
    points_df = points_df[front_90_mask]
    df_len = len(points_df)
    # Reindex by row number
    points_df.index = np.arange(df_len)

    # distance to the point is one of the interesting features
    points_df['distance'] = np.linalg.norm(np.vstack((points_df.x.values,
                                                      points_df.y.values,
                                                      points_df.z.values)), axis=0)

    # We want to divide front 90 degrees into number of bins equal to width
    bin_size = 90 / RV_WIDTH
    points_df['azimuth_bin'] = np.digitize(x=points_df['azimuth'].values - 45,  # right-most value is 45
                                           bins=np.arange(0, 90, bin_size))

    # artificial "point" which will represent values for missing points
    points_df.loc[df_len] = (0, 0, 0, 0, 0, 0, 0, 0, 1)

    # Finally construct the Range View
    # First, we construct 7 channels: x, y, height(z), intensity, aziumth, distance
    range_view = np.zeros((RV_HEIGHT, RV_WIDTH, 7))
  
    # row numbers of points which have minimal distance in their groups

    idx_min = points_df.groupby(['ring_index', 'azimuth_bin']) \
        .distance.idxmin() \
        .unstack(fill_value=df_len).stack().values
    points_df = points_df.loc[idx_min]

    # class 0 is for noised
    point_labels = pd.get_dummies(points_df['class']).T.reindex(range(LABEL_NUMBER)).T.fillna(0)
    points_df.drop(['class', 'ring_index', 'azimuth_bin'], inplace=True, axis=1)
    try:
        range_view[:, :, :6] = points_df.values.reshape((RV_HEIGHT, RV_WIDTH, 6))
        point_labels = point_labels.values.reshape((RV_HEIGHT, RV_WIDTH, LABEL_NUMBER))

        # 7th channel is a flag which shows whether there is a point or not.
        # If distance == azimuth == ring_index == 0 => no point
        range_view[:, :, 6] = (range_view[:, :, :].sum(axis=2) != 0).astype(int)

    except ValueError:
        return np.zeros((RV_HEIGHT, RV_WIDTH, 7)), np.zeros((RV_HEIGHT, RV_WIDTH, LABEL_NUMBER))

    # need to reflect x and y, so it matches camera view
    return range_view[::-1, ::-1, :], point_labels[::-1, ::-1]


def sample_to_rangeview(sample: dict) -> np.array:
    """
        Transform sample's Lidar Point Cloud to Range View.

        Range View is an image of size H x W x 5,
        where H - number of lasers, W - number of discretized azimuth bins.
        Each of 5 channels is responsible for a single interesting feature:
        range, height, azimuth, intensity and flag indicating whether there is a point or not

        Lookup into each of feature matrices looks like matrix[laser_number][azimuth_bin]
        
        returns: the range_view, coordinates of points which got to the rv, labels of those points
    """

    sample_data_token = sample['data']['LIDAR_TOP']

    my_sample_lidar_data = NUSCENES.get('sample_data', sample_data_token)

    lidarseg_labels_filename = join(NUSCENES.dataroot,
                                    NUSCENES.get('lidarseg', sample_data_token)['filename'])

    # loading directly from files to perceive the ring_index information
    points_raw = np.fromfile(DATASET_PATH + my_sample_lidar_data["filename"], dtype=np.float32).reshape((-1, 5))
    point_labels = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)

    return pcl_to_rangeview(points_raw, point_labels)
