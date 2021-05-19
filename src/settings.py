from nuscenes import NuScenes

DATASET_PATH = './data/'
NUSCENES = NuScenes(version='v1.0-trainval', dataroot=DATASET_PATH, verbose=True)
RV_WIDTH = 256
RV_HEIGHT = 32  # there are 32 lasers in NUSCENES' LiDAR
LABEL_NUMBER = 32
