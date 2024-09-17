import os
from PIL import Image


def add_padding_and_resize(image, target_width, target_height):
    original_width, original_height = image.size

    # Calculate the aspect ratio
    aspect_ratio = original_width / original_height

    # Determine the new size while maintaining the aspect ratio
    if aspect_ratio > (target_width / target_height):  # Wider image
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:  # Taller image
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Create a new image with the target size and white background
    new_image = Image.new("RGB", (target_width, target_height), (255, 255, 255))

    # Paste the resized image onto the new image, centered
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def process_images(input_folder, output_folder, target_width=720, target_height=1280):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)

        # Only process image files
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            with Image.open(file_path) as img:
                processed_img = add_padding_and_resize(img, target_width, target_height)

                # Save as .jpg
                new_file_name = file_name.rsplit('.', 1)[0] + ".jpg"
                output_file_path = os.path.join(output_folder, new_file_name)
                processed_img.save(output_file_path)


if __name__ == "__main__":
    input_folder = "/workspace/ai-tailer-detectron/ai-tailer-detectron/input_data/images/front"  # Replace with your input folder path
    output_folder = "/workspace/ai-tailer-detectron/ai-tailer-detectron/output_data/images/front"  # Replace with your output folder path
    process_images(input_folder, output_folder)
