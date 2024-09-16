# Step1: Convert The CSV To Json
csv_file_path="/ai-tailer-detectron/input_data/data_kps.csv"
json_file_path="/ai-tailer-detectron/input_data/data_kps.json"
python csv2json_conversion.py --input_csv "$csv_file_path" --output_json "$json_file_path"

## ===================================================================================== ##

# Step2: Generate Masks For Input Data
images_folder_path="/ai-tailer-detectron/input_data/image_processed"
output_mask_folder_path="/ai-tailer-detectron/output_data/masks"
python human_segmentation.py --input_images_folder "$images_folder_path" --output_mask_folder "$output_mask_folder_path"

## ===================================================================================== ##

# Step3: Create Augmented Data
input_backgrounds_folder="/ai-tailer-detectron/input_data/keypoint-detection/dataset/backgrounds/bg_sm"
output_augmented_dir="/ai-tailer-detectron/input_data/data_for_training"
# input_backgrounds_folder="/media/nithin/17b8fbe0-a0a3-46d6-bb0e-d8bebe7bcbd0/experimental/KeypointsRepos/keypoint-detection/dataset/backgrounds/bg_sm"
# output_augmented_dir="/home/nithin/Downloads/test/data_for_training"
num_augs=20

python create_augmentation_keypoints.py \
    --input_images_folder "$images_folder_path" \
    --input_mask_folder "$output_mask_folder_path" \
    --input_backgrounds_folder "$input_backgrounds_folder" \
    --input_json_path "$json_file_path" \
    --output_augmented_dir "$output_augmented_dir" \
    --num_augs "$num_augs"

## ===================================================================================== ##
