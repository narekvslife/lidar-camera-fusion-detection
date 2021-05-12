from nuscenes import NuScenes

DATASET_PATH = '../../data/sets/nuscenes/MINI/'
NUSCENES = NuScenes(version='v1.0-mini', dataroot=DATASET_PATH, verbose=True)
RV_WIDTH = 128
RV_HEIGHT = 32  # there are 32 lasers in NUSCENES' LiDAR
LABEL_NUMBER = 32
