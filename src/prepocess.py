import numpy as np
import pandas as pd


from .settings import DATASET_PATH, NUSCENES


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

    my_sample_lidar_data = NUSCENES.get('sample_data', sample['data']['LIDAR_TOP'])

    # loading directly from files to preceive the ring_index information
    scan = np.fromfile(DATASET_PATH + my_sample_lidar_data["filename"], dtype=np.float32)
    raw_points = scan.reshape((-1, 5))

    points_df = pd.DataFrame(raw_points, columns=['x', 'y', 'z', 'intensity', 'ring_index'])
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
    # Add adrtificial row

    # distance to the point is one of the interesting features
    points_df['distance'] = (points_df.x ** 2 + points_df.y ** 2 + points_df.z ** 2) ** 1 / 2

    # We want to divide front 90 degrees into number of bins equal to width
    bin_size = 90 / width
    points_df['azimuth_bin'] = np.digitize(x=points_df['azimuth'].values - 45,  # right-most value is 45
                                           bins=np.arange(0, 90, bin_size))

    # artificial "point" which will represent values for missing points
    points_df.loc[df_len] = (0, 0, 0, 0, 0, 0, 0, 1)

    # Finally construct the Range View image
    # First, we construct 4 channels: height, intensity, aziumth, distance
    image = np.zeros((height, width, 5))
    #     try:
    # row numbers of points which have minimal distance in their groups
    idx_min = points_df.groupby(['ring_index', 'azimuth_bin']).distance.idxmin().unstack(
        fill_value=df_len).stack().values
    points_df.drop(['ring_index', 'azimuth_bin'], inplace=True, axis=1)
    points_features = points_df.loc[idx_min].values
    points_features = points_features.reshape((height, width, 6))

    # we will need coordinates of a point in each cell later in LaserNet
    point_coordinates = points_features[:, :, :2]

    # here are the 4 channels: distance, azimuth, reflectance, and height
    image[:, :, :4] = points_features[:, :, 2:]

    # 5th channel is a flag which shows whether there is a point or not.
    # If distance == azimuth == ring_index == 0 => no point
    image[:, :, 4] = (image[:, :, :].sum(axis=2) != 0).astype(int)

    assert image.shape[: 2] == point_coordinates.shape[: 2]
    # need to reflect x and y, so it matches camera view
    image = image[::-1, ::-1, :]
    point_coordinates = point_coordinates[::-1, ::-1, :]
    #     except ValueError:
    #         print(sample['token'])

    return image, point_coordinates
