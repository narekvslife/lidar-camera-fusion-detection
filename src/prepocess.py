import numpy as np
import pandas as pd
from os.path import join

from .settings import DATASET_PATH, NUSCENES
from scipy.spatial.transform import Rotation


def compute_absolute_coordinates(ego_translation: np.array,
                                 ego_rotation_q: np.array,
                                 relative_coordinates: np.array):
    assert len(ego_rotation_q) == 4
    assert len(ego_translation) == 3

    R = Rotation.from_quat(ego_rotation_q).as_matrix()[:2, :2]
    relative_coordinates_unsq = np.expand_dims(relative_coordinates, 3)

    #     print(R.shape, relative_coordinates_unsq.shape)
    absolute_coordinates = ego_translation[:2] + (R @ relative_coordinates_unsq).squeeze(3)
    return absolute_coordinates


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

    # loading directly from files to preceive the ring_index information
    points_raw = np.fromfile(DATASET_PATH + my_sample_lidar_data["filename"], dtype=np.float32).reshape((-1, 5))
    point_labeles = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)

    points_df = pd.DataFrame(points_raw, columns=['x', 'y', 'z', 'intensity', 'ring_index'])
    points_df['class'] = point_labeles
    points_df['azimuth'] = np.degrees(np.arctan(points_df['x'] / points_df['y']))

    # Transform front azimuth to be in range from 0 to 180
    mask_front = (points_df.y >= 0)
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
    points_df['distance'] = (points_df.x ** 2 + points_df.y ** 2 + points_df.z ** 2) ** 1 / 2

    # We want to divide front 90 degrees into number of bins equal to width
    bin_size = 90 / width
    points_df['azimuth_bin'] = np.digitize(x=points_df['azimuth'].values - 45,  # right-most value is 45
                                           bins=np.arange(0, 90, bin_size))

    # artificial "point" which will represent values for missing points
    points_df.loc[df_len] = (0, 0, 0, 0, 0, 0, 0, 0, 1)

    # Finally construct the Range View image
    # First, we construct 4 channels: height, intensity, aziumth, distance
    image = np.zeros((height, width, 5))
    try:
        # we only leave the points which have minimal distance in their groups
        idx_min = points_df.groupby(['ring_index', 'azimuth_bin']) \
            .distance.idxmin() \
            .unstack(fill_value=df_len).stack().values
        points_df = points_df.loc[idx_min]

        point_labeles = points_df['class'].values.reshape((height, width))

        points_df.drop(['ring_index', 'azimuth_bin', 'class'], inplace=True, axis=1)

        points_features = points_df.values.reshape((height, width, 6))

        # during LaserNet inference, we will need absolute coordinates of a point in each cell
        relative_point_coordinates = points_features[:, :, :2]
        ego = NUSCENES.get('ego_pose', my_sample_lidar_data['ego_pose_token'])
        absolute_coordinates = compute_absolute_coordinates(ego_translation=ego['translation'],
                                                            ego_rotation_q=ego['rotation'],
                                                            relative_coordinates=relative_point_coordinates)

        # here are the 4 channels: distance, azimuth, reflectance, and height
        image[:, :, :4] = points_features[:, :, 2:]

        # 5th channel is a flag which shows whether there is a point or not.
        # If distance == azimuth == ring_index == 0 => no point
        image[:, :, 4] = (image[:, :, :].sum(axis=2) != 0).astype(int)

        assert image.shape[: 2] == absolute_coordinates.shape[: 2]

    except ValueError:
        print(sample['token'])

    # need to reflect x and y, so it matches camera view
    return image[::-1, ::-1, :], absolute_coordinates[::-1, ::-1, :], point_labeles[::-1, ::-1]
