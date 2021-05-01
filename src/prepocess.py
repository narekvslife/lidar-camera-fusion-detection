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

    # distance to the point is one of the interesting features
    points_df['distance'] = (points_df.x ** 2 + points_df.y ** 2 + points_df.z ** 2) ** 1 / 2

    # We want to divide front 90 degrees into number of bins equal to width
    bin_size = 90 / width
    points_df['azimuth_bin'] = np.digitize(x=points_df['azimuth'].values - 45,  # right-most value is 45
                                           bins=np.arange(0, 90, bin_size))

    # Finally construct the Range View image
    # First, we construct 4 channels: height, intensity, aziumth, distance
    image = np.zeros((height, width, 5))
    try:
        # create a table with index [ring_index, azimuth_bin]
        # and values being the distance of a closest point in that group
        min_dist_cross_tab = pd.crosstab(points_df.azimuth_bin,
                                         points_df.ring_index,
                                         values=points_df.distance,
                                         aggfunc='min')
        points_features = []

        for ring_index in range(0, height):
            for azimuth_bin in range(1, width + 1):

                dist_value = min_dist_cross_tab[ring_index][azimuth_bin]
                if np.isnan(dist_value):
                    # if there were no points in [ring_index, azimuth_bin] group, we fill feature values with 0
                    feature_values = np.zeros((1, 6))
                else:
                    # here we find the clossest point of a group
                    mask = (points_df['azimuth_bin'] == azimuth_bin) \
                           & (points_df['ring_index'] == ring_index) \
                           & (points_df['distance'] == dist_value)
                    # if there are multiple values, we take any of them
                    feature_values = points_df[mask].drop(['ring_index', 'azimuth_bin'], axis=1).values[0]

                points_features.append(feature_values)

        points_features = np.vstack(points_features).reshape((height, width, 6))

        # we will need coordinates of a point in each cell later in LaserNet
        point_coordinates = points_features[:, :, :2]

        # here are the 4 channels: distance, azimuth, reflectance, and height
        image[:, :, :4] = points_features[:, :, 2:]

        # 5th channel is a flag which shows whether there is a point or not.
        # If distance == azimuth == ring_index == 0 => no point
        image[:, :, 4] = (image[:, :, :].sum(axis=2) != 0).astype(int)

        assert image.shape[: 2] == point_coordinates.shape[: 2]
        # need to reflect x and y, so it matches camera view
        image = image[::-1, ::-1, :].T
        point_coordinates = point_coordinates[::-1, ::-1, :].T
    except ValueError:
        print(sample['token'])

    return image, point_coordinates
