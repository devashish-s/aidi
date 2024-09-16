import pandas as pd
import json
import argparse


def convert_csv_to_json(csv_file_path, output_json_path):
    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # Initialize the structure for the JSON output
    json_output = {
        "info": {
            "description": "Converted dataset",
            "url": "",
            "version": "1.0",
            "year": 2024,
            "contributor": "User",
            "date_created": "2024/09/09"
        },
        "licenses": [
            {
                "url": "https://creativecommons.org/licenses/by/2.0/",
                "id": 1,
                "name": "Attribution Generic License"
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "box",
                "keypoints": ["Head", "LeftShoulder", "RightShoulder", "LeftChest", "RightChest",
                              "LeftWaist", "RightWaist", "LeftHighHip", "RightHighHip", "LeftHip",
                              "RightHip", "LeftFoot", "RightFoot"],
                "skeleton": []
            }
        ],
        "images": [],
        "annotations": []
    }

    # Group the data by filename to handle each image separately
    grouped = df.groupby('filename')

    annotation_id = 1

    for filename, group in grouped:
        # Create image entry
        image_entry = {
            "id": annotation_id,
            "file_name": filename,
            "width": 720,  # Assuming a default width as it's not provided in CSV
            "height": 1280,  # Assuming a default height as it's not provided in CSV
            "license": 1,
            "date_captured": ""
        }

        # Add the image entry to the images list
        json_output["images"].append(image_entry)

        # Prepare keypoints and bounding box
        keypoints = []
        for _, row in group.iterrows():
            region_shape = json.loads(row['region_shape_attributes'])
            keypoints.extend([region_shape['cx'], region_shape['cy'], 2])  # 2 indicates visibility

        # Calculate bounding box
        all_x = [json.loads(row['region_shape_attributes'])['cx'] for _, row in group.iterrows()]
        all_y = [json.loads(row['region_shape_attributes'])['cy'] for _, row in group.iterrows()]
        x_min, y_min = min(all_x), min(all_y)
        width, height = max(all_x) - x_min, max(all_y) - y_min
        bbox = [x_min, y_min, width, height]

        # Create annotation entry
        annotation_entry = {
            "image_id": annotation_id,
            "category_id": 1,
            "keypoints": keypoints,
            "num_keypoints": len(group),
            "id": annotation_id,
            "bbox": bbox,
            "bbox_mode": 1
        }

        # Add the annotation entry to the annotations list
        json_output["annotations"].append(annotation_entry)

        annotation_id += 1

    # Save the resulting JSON file
    with open(output_json_path, 'w') as f:
        json.dump(json_output, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSV and JSON file paths.')
    parser.add_argument('--input_csv', required=True, help='Path to the CSV file')
    parser.add_argument('--output_json', required=True, help='Path to the JSON file')

    args = parser.parse_args()

    # Call your function with the provided paths
    convert_csv_to_json(args.input_csv, args.output_json)
    print("Json Conversion Success")
