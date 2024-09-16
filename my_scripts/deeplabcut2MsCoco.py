import pandas as pd
import json

# Load the CSV file
csv_file_path = '/home/nithin/Downloads/side_CollectedData_front_pose.csv'
data = pd.read_csv(csv_file_path)

# Define keypoints names
keypoints_names = [
    "Head", "LeftShoulder", "RightShoulder", "LeftChest", "RightChest",
    "LeftWaist", "RightWaist", "LeftHighHip", "RightHighHip",
    "LeftHip", "RightHip", "LeftFoot", "RightFoot"
]

# Define categories
categories = [{
    "id": 1,
    "name": "box",
    "keypoints": keypoints_names,
    "skeleton": []
}]

# Initialize images and annotations
images_unique = []
annotations_unique = []

# Use a set to track added images to avoid duplication
added_images = set()

# Assuming the images are named sequentially and have the same dimensions
width, height = 720, 1280  # Update with actual dimensions if different

# Process each image's data
for idx, row in data.iterrows():
    if row['scorer'] == 'labeled-data':
        file_name = row['Unnamed: 2']
        if file_name not in added_images:
            added_images.add(file_name)
            image_id = len(images_unique)
            images_unique.append({
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
                "license": 1,
                "date_captured": ""
            })

            keypoints = []
            for i in range(len(keypoints_names)):
                x_col = f'front_pose.{i*2}' if i > 0 else 'front_pose'
                y_col = f'front_pose.{i*2+1}' if i > 0 else 'front_pose.1'
                x = row[x_col]  # 'x' coordinate
                y = row[y_col]  # 'y' coordinate
                visibility = 2 if pd.notna(x) and pd.notna(y) else 0
                keypoints.extend([float(x), float(y), visibility])

            # Calculate bounding box [x_min, y_min, width, height]
            x_coords = keypoints[::3]
            y_coords = keypoints[1::3]
            x_min, y_min = min(x_coords), min(y_coords)
            bbox_width, bbox_height = max(x_coords) - x_min, max(y_coords) - y_min

            annotations_unique.append({
                "image_id": image_id,
                "category_id": 1,
                "keypoints": keypoints,
                "num_keypoints": len(keypoints_names),
                "id": image_id,
                "bbox": [x_min, y_min, bbox_width, bbox_height],
                "bbox_mode": 1
            })

# Compile the final JSON structure with unique images
mscoco_json_unique = {
    "info": {
        "description": "Converted dataset",
        "url": "",
        "version": "1.0",
        "year": 2024,
        "contributor": "User",
        "date_created": "2024/08/30"
    },
    "licenses": [{
        "url": "https://creativecommons.org/licenses/by/2.0/",
        "id": 1,
        "name": "Attribution Generic License"
    }],
    "categories": categories,
    "images": images_unique,
    "annotations": annotations_unique
}

# Save the JSON to a file
json_file_path_unique = 'side_pose.json'
with open(json_file_path_unique, 'w') as json_file:
    json.dump(mscoco_json_unique, json_file)

print(f"JSON file saved to {json_file_path_unique}")
