import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.model_zoo import get_config_file, get_checkpoint_url


# Define keypoint names
keypoint_names = [
    "Head", "LeftShoulder", "RightShoulder", "LeftChest", "RightChest",
    "LeftWaist", "RightWaist", "LeftHighHip", "RightHighHip",
    "LeftHip", "RightHip", "LeftFoot", "RightFoot"
]

keypoint_flip_map = [
    ("LeftShoulder", "RightShoulder"),
    ("LeftChest", "RightChest"),
    ("LeftWaist", "RightWaist"),
    ("LeftHighHip", "RightHighHip"),
    ("LeftHip", "RightHip"),
    ("LeftFoot", "RightFoot")
]

pose_name = "front"
train_name = f"train_{pose_name}_keypoints"
val_name = f"val_{pose_name}_keypoints"


# Provide Json Path
train_json_path = "/workspace/ai-tailer-detectron/input_data/data_for_training/augmented_data/train_augmented_keypoints.json"
val_json_path = "/workspace/ai-tailer-detectron/input_data/data_for_training/augmented_data/val_augmented_keypoints.json"

# Specify train and val Images dir
train_data_dir = "/workspace/ai-tailer-detectron/input_data/data_for_training/augmented_data/train_data"
val_data_dir = "/workspace/ai-tailer-detectron/input_data/data_for_training/augmented_data/train_data"

# Register the datasets
register_coco_instances(train_name, {},
                        train_json_path,
                        train_data_dir)
register_coco_instances(val_name, {},
                        val_json_path,
                        val_data_dir)

# Add keypoint names and flip map to the metadata
MetadataCatalog.get(train_name).keypoint_names = keypoint_names
MetadataCatalog.get(val_name).keypoint_names = keypoint_names
MetadataCatalog.get(train_name).keypoint_flip_map = keypoint_flip_map
MetadataCatalog.get(val_name).keypoint_flip_map = keypoint_flip_map

# Load configurations
cfg = get_cfg()
cfg.merge_from_file(get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (train_name,)
cfg.DATASETS.TEST = (val_name,)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 12
cfg.SOLVER.BASE_LR = 0.00025

cfg.SOLVER.CHECKPOINT_PERIOD = 2000
cfg.SOLVER.MAX_ITER = 20000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Number of classes (excluding background)
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 13  # Number of keypoints

# Set input size to 480x640
cfg.INPUT.MIN_SIZE_TRAIN = (640,)
cfg.INPUT.MAX_SIZE_TRAIN = 480
cfg.INPUT.MIN_SIZE_TEST = 640
cfg.INPUT.MAX_SIZE_TEST = 480

cfg.OUTPUT_DIR = f"./output/{pose_name}_pose_model"

# Train the model
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
