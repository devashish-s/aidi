import glob
import cv2


imagePaths = glob.glob("../dataset/front_pose_mask/*")

for imagePath in imagePaths:
    image_np = cv2.imread(imagePath)


    image_np_blur = cv2.GaussianBlur(image_np, (3,3), 0)

    cv2.imwrite(imagePath, image_np_blur)

