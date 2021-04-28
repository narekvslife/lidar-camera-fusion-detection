from nuscenes import NuScenes

DATASET_PATH = '../../data/sets/nuscenes/MINI/'
NUSCENES = NuScenes(version='v1.0-mini', dataroot=DATASET_PATH, verbose=True)
