import argparse
import os
import torch
from torchvision import models, transforms
from PIL import Image, ImageFilter
import numpy as np
from tqdm import tqdm

# Load a pre-trained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# Define the image transformation
preprocess = transforms.Compose([
    transforms.Resize(520),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def generate_masks(input_folder, output_mask_folder):
    os.makedirs(output_mask_folder, exist_ok=True)

    print("Generating Masks...")
    # Process each image in the input folder
    for image_name in tqdm(os.listdir(input_folder)):
        image_path = os.path.join(input_folder, image_name)
        input_image = Image.open(image_path)

        # Save the original image size
        original_size = input_image.size

        # Apply preprocessing
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

        # Perform the segmentation
        with torch.no_grad():
            output = model(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        # Create a binary mask where the human is labeled as 1
        mask = (output_predictions == 15).byte().cpu().numpy()  # Class 15 corresponds to 'person'

        # Convert mask to an image format (0-255)
        mask_image = Image.fromarray(mask * 255)

        # Resize the mask to match the original image size
        mask_image = mask_image.resize(original_size, Image.NEAREST)

        gBlur = ImageFilter.GaussianBlur(radius=2)  # Adjust radius as needed
        mask_image = mask_image.filter(gBlur)

        # Save the mask
        mask_name = os.path.splitext(image_name)[0] + ".png"
        mask_path = os.path.join(output_mask_folder, mask_name)
        mask_image.save(mask_path)

        # print(f"Processed and saved mask for {image_name} at {mask_path}")

    print("Generated mask images.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Mask images.')
    parser.add_argument('--input_images_folder', required=True)
    parser.add_argument('--output_mask_folder', required=True)
    args = parser.parse_args()

    # Ensure the output folder exists

    # Call your function with the provided paths
    generate_masks(args.input_images_folder, args.output_mask_folder)