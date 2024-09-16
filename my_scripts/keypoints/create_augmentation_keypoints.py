import argparse
import os
import pdb

import cv2
import json
import random
import numpy as np
from tqdm import tqdm


def save_image(image, filepath):
    success = cv2.imwrite(filepath, image)
    if not success:
        raise Exception(f"Failed to save image at {filepath}")


def alpha_blending(foreground_np, background_np, mask, alpha=0.7):
    try:
        mask_transformed = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
    except:
        mask_transformed = mask.astype(float) / 255.0

    background_np = cv2.resize(background_np, foreground_np.shape[1::-1])
    return (foreground_np * mask_transformed + background_np * (1 - mask_transformed)).astype("uint8")


def apply_augmentations(image, keypoints):
    angle = random.uniform(-5, 5)
    M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    for i in range(0, len(keypoints), 3):
        x, y = keypoints[i], keypoints[i + 1]

        if x is None or y is None:
            keypoints[i], keypoints[i + 1] = 0, 0
            x, y  = 0, 0

        if x != 0 and y != 0:
            x_new = M[0, 0] * x + M[0, 1] * y + M[0, 2]
            y_new = M[1, 0] * x + M[1, 1] * y + M[1, 2]
            keypoints[i], keypoints[i + 1] = x_new, y_new

    max_trans = 10  # Maximum pixels to translate
    tx = random.randint(-max_trans, max_trans)
    ty = random.randint(-max_trans, max_trans)
    trans_M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(rotated, trans_M, (image.shape[1], image.shape[0]))

    for i in range(0, len(keypoints), 3):
        if keypoints[i] != 0 and keypoints[i + 1] != 0:
            keypoints[i] += tx
            keypoints[i + 1] += ty

    beta = random.uniform(-20, 20)  # brightness change [-20, 20]
    brightened = cv2.convertScaleAbs(translated, alpha=1, beta=beta)

    return brightened, keypoints


def process_image(foreground_img, background_img, mask_img, annotation, output_dir, image_id):
    new_annotations = []
    new_images = []

    # Random choice: Augmentation or Blending
    if random.random() < 0.5:

        image_id += 1
        # Augmentation
        augmented_img, aug_keypoints = apply_augmentations(foreground_img, annotation['keypoints'].copy())
        augmented_path = f"{output_dir}/augmented_{image_id}.jpg"
        save_image(augmented_img, augmented_path)

        if os.path.exists(augmented_path):
            new_annotations.append({
                "image_id": image_id,
                "category_id": annotation['category_id'],
                "keypoints": aug_keypoints,
                "num_keypoints": annotation['num_keypoints'],
                "id":image_id,
                "bbox": annotation["bbox"]
            })
            new_images.append({
                "id": image_id,
                "file_name": augmented_path,
                "width": foreground_img.shape[1],
                "height": foreground_img.shape[0],
                "license": 1,
                "date_captured": annotation.get('date_captured', '')
            })

    elif random.random() < 0.5:
        image_id += 1

        # Blending
        blended_img = alpha_blending(foreground_img, background_img, mask_img)
        blended_path = f"{output_dir}/augmented_{image_id}.jpg"
        save_image(blended_img, blended_path)

        if os.path.exists(blended_path):
            new_annotations.append({
                "image_id": image_id,
                "category_id": annotation['category_id'],
                "keypoints": annotation['keypoints'].copy(),
                "num_keypoints": annotation['num_keypoints'],
                "id": image_id,
                "bbox": annotation["bbox"]
            })
            new_images.append({
                "id": image_id,
                "file_name": blended_path,
                "width": blended_img.shape[1],
                "height": blended_img.shape[0],
                "license": 1,
                "date_captured": annotation.get('date_captured', '')
            })

            # Random choice: Augmentation or Blending
        if random.random() < 0.5:

            image_id += 1
            # Augmentation
            augmented_img, aug_keypoints = apply_augmentations(blended_img, annotation['keypoints'].copy())
            augmented_path = f"{output_dir}/augmented_{image_id}.jpg"
            save_image(augmented_img, augmented_path)

            if os.path.exists(augmented_path):
                new_annotations.append({
                    "image_id": image_id,
                    "category_id": annotation['category_id'],
                    "keypoints": aug_keypoints,
                    "num_keypoints": annotation['num_keypoints'],
                    "id": image_id,
                    "bbox": annotation["bbox"]
                })
                new_images.append({
                    "id": image_id,
                    "file_name": augmented_path,
                    "width": blended_img.shape[1],
                    "height": blended_img.shape[0],
                    "license": 1,
                    "date_captured": annotation.get('date_captured', '')
                })

    return new_annotations, new_images, image_id


def split_dataset(keypoints_data, train_ratio=0.9):
    """
    Splits the dataset into training and validation sets.

    Args:
    keypoints_data (dict): The dataset containing 'images' and 'annotations'.
    train_ratio (float): Ratio of data to be used for training. Defaults to 0.8 (80% train, 20% val).

    Returns:
    train_images, train_annotations, val_images, val_annotations: Split datasets.
    """
    # Get the total number of images
    total_images = len(keypoints_data['images'])

    # Create a list of indices and shuffle them
    indices = list(range(total_images))
    random.shuffle(indices)

    # Calculate the split index
    split_idx = int(total_images * train_ratio)

    # Split images and annotations based on shuffled indices
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # Get corresponding images and annotations for training and validation sets
    train_images = [keypoints_data['images'][i] for i in train_indices]
    train_annotations = [keypoints_data['annotations'][i] for i in train_indices]

    val_images = [keypoints_data['images'][i] for i in val_indices]
    val_annotations = [keypoints_data['annotations'][i] for i in val_indices]

    return train_images, train_annotations, val_images, val_annotations


def main(images_folder, masks_folder, backgrounds_folder, json_path, output_augmented_dir, num_augs):
    with open(json_path) as f:
        keypoints_data = json.load(f)

    # Split dataset into training and validation sets
    train_images, train_annotations, val_images, val_annotations = split_dataset(keypoints_data,
                                                                                 train_ratio=0.9)
    for mode in ["train", "val"]:
        output_dir = os.path.join(output_augmented_dir, f"augmented_data/{mode}_data")
        os.makedirs(output_dir, exist_ok=True)

        target_count_ = int(num_augs * 0.9) if mode == "train" else int(num_augs * 0.1)

        generated_count = 0
        final_images = []
        final_annotations = []

        # Set the correct images and annotations based on the mode
        if mode == "train":
            images = train_images
            annotations = train_annotations
        else:
            images = val_images
            annotations = val_annotations


        while generated_count < target_count_:
            for image_info, annotation in tqdm(zip(images, annotations), desc="Generating images"):
                if generated_count >= target_count_:
                    break

                image_filename = image_info['file_name']  # Use the image_id to find the correct image filename in the images folder
                foreground_img = cv2.imread(os.path.join(images_folder, f"{image_filename}"))
                if foreground_img is None:
                    print(f"Image {image_filename}.jpg not found, skipping.")
                    continue

                mask_img = cv2.imread(os.path.join(masks_folder, f"{os.path.basename(image_filename).replace('.jpg', '.png')}"), cv2.IMREAD_GRAYSCALE)
                if mask_img is None:
                    print(f"Mask for {image_filename}.png not found, skipping.")
                    continue

                background_img = cv2.imread(
                    random.choice([os.path.join(backgrounds_folder, f) for f in os.listdir(backgrounds_folder)]))
                if background_img is None:
                    print("Background image not found, skipping.")
                    continue

                new_annots, new_images, generated_count = process_image(foreground_img, background_img, mask_img, annotation, output_dir,
                                                       generated_count)
                final_annotations.extend(new_annots)
                final_images.extend(new_images)


        # Save the new JSON with updated annotations
        keypoints_data['images'] = final_images
        keypoints_data['annotations'] = final_annotations

        with open(os.path.join(output_dir, f'../{mode}_augmented_keypoints.json'), 'w') as f:
            json.dump(keypoints_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Augmented Data.')
    parser.add_argument('--input_images_folder', required=True)
    parser.add_argument('--input_mask_folder', required=True)
    parser.add_argument('--input_backgrounds_folder', required=True)
    parser.add_argument('--input_json_path', required=True)
    parser.add_argument('--output_augmented_dir', required=True)
    parser.add_argument('--num_augs', default=2200, type=int, required=False)
    args = parser.parse_args()


    images_folder = args.input_images_folder
    masks_folder = args.input_mask_folder
    backgrounds_folder = args.input_backgrounds_folder
    json_path = args.input_json_path
    num_augs = args.num_augs

    main(images_folder, masks_folder, backgrounds_folder, json_path, args.output_augmented_dir, num_augs)
