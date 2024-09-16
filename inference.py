import glob

from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_config_file, get_checkpoint_url
import os, cv2, pdb

cfg = get_cfg()
cfg.merge_from_file(get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train_side_keypoints",)
cfg.DATASETS.TEST = ("val_side_keypoints",)
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 12
cfg.SOLVER.BASE_LR = 0.00025

cfg.SOLVER.CHECKPOINT_PERIOD = 2000
cfg.SOLVER.MAX_ITER = 10000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Number of classes (excluding background)
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 13  # Number of keypoints

# Set input size to 480x640
cfg.INPUT.MIN_SIZE_TRAIN = (640,)
cfg.INPUT.MAX_SIZE_TRAIN = 480
cfg.INPUT.MIN_SIZE_TEST = 640
cfg.INPUT.MAX_SIZE_TEST = 480

cfg.OUTPUT_DIR = "./output/side_pose_model"
cfg.MODEL.DEVICE = "cpu"


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)


imagePaths = glob.glob("/home/nithin/Desktop/detectron2/side/*")

output_path = "/home/nithin/Desktop/detectron2/side_vis/"

print("totalImages: ", len(imagePaths))
for imagePath in imagePaths:
    im = cv2.imread(imagePath)[:,:,::-1]
    outputs = predictor(im)


    # Assuming the inference output is stored in a variable named `outputs`
    instances = outputs["instances"]
    keypoints = instances.pred_keypoints if instances.has("pred_keypoints") else None



    if keypoints is not None:
        # Create a visualizer instance
        visualizer = Visualizer(im, scale=1.0)

        # Draw keypoints
        visualizer = visualizer.draw_instance_predictions(instances)

        # Get the output image with keypoints drawn
        output_image = visualizer.get_image()[:, :, ::-1]

        # Display or save the image
        # cv2.imshow("Keypoints Image", cv2.resize(output_image, None, fx=0.7, fy=0.7))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        # Optionally, save the image to a file
        cv2.imwrite(os.path.join(output_path, os.path.basename(imagePath)), output_image)
    else:
        print("No keypoints found in the output.")


