import os
COMPUTER_NAME = os.environ['COMPUTERNAME']
print("Computer: ", COMPUTER_NAME)

WORKER_POOL_SIZE = 1

TARGET_VOXEL_MM = 1.00
MEAN_PIXEL_VALUE_NODULE = 41
LUNA_SUBSET_START_INDEX = 0
SEGMENTER_IMG_SIZE = 320

BASE_DIR_SSD = "D:/jupyter-notebook/LungCancerPredict/"
BASE_DIR = "D:/jupyter-notebook/LungCancerPredict/"
EXTRA_DATA_DIR = "resources/"
NDSB3_RAW_SRC_DIR = BASE_DIR + "original/ndsb_raw/"
LUNA16_RAW_SRC_DIR = BASE_DIR + "original/luna_raw/"

NDSB3_EXTRACTED_IMAGE_DIR = BASE_DIR_SSD + "extracted/ndsb3_extracted_images/"
LUNA16_EXTRACTED_IMAGE_DIR = BASE_DIR_SSD + "extracted/luna16_extracted_images/"
NDSB3_NODULE_DETECTION_DIR_CNN = BASE_DIR_SSD + "extracted/ndsb3_nodule_predictions_CNN/"
NDSB3_NODULE_DETECTION_DIR_RNN = BASE_DIR_SSD + "extracted/ndsb3_nodule_predictions_RNN/"
LUNA16_NODULE_DETECTION_DIR_CNN = BASE_DIR_SSD + "extracted/luna16_nodule_predictions_CNN/"
LUNA16_NODULE_DETECTION_DIR_RNN = BASE_DIR_SSD + "extracted/luna16_nodule_predictions_RNN/"

