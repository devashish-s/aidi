import json


def add_bboxes_to_annotations(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    for annotation in data['annotations']:
        keypoints = annotation['keypoints']
        x_coords = keypoints[0::3]  # Extract x coordinates
        y_coords = keypoints[1::3]  # Extract y coordinates

        # Calculate bounding box (xmin, ymin, width, height)
        xmin = min(x_coords)
        xmax = max(x_coords)
        ymin = min(y_coords)
        ymax = max(y_coords)
        width = xmax - xmin
        height = ymax - ymin

        annotation['bbox'] = [xmin, ymin, width, height]
        annotation['bbox_mode'] = 1  # Assuming COCO format (XYWH_ABS)

    # Save the updated JSON
    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)


# Paths to your dataset JSON files
train_json_path = "/workspace/ai-tailer-detectron/ai-tailer-detectron/input_data/data_for_training/augmented_data/train_augmented_keypoints.json"
#train_json_path = '/home/nithin/Downloads/SIZE/via_project_9Sep2024_11h34m_coco.json'
# val_json_path = '/home/nithin/Desktop/detectron2/detectron2/dataset/augmented_side/val_augmented_side_keypoints.json'

# Add bounding boxes to both train and validation datasets
add_bboxes_to_annotations(train_json_path)
# add_bboxes_to_annotations(val_json_path)

