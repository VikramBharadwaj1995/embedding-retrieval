import cv2
import numpy as np
#import torch
import sys
from PIL import Image
import os


# Load YOLO model
net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")

# Load class names
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]





sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
)



# Specify the directory containing the images
image_directory = 'images'  # Replace with the path to your image directory

# Get the list of image file names in the directory
image_files = os.listdir(image_directory)

# Loop through the image files and resize them
for file_name in image_files:

    # Open the image
    image_path = os.path.join(image_directory, file_name)
    print(image_path)
    if not file_name.endswith('.jpeg') and not file_name.endswith('.jpg'):
        continue
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # Preprocess the image
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Perform object detection
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Filter and draw bounding boxes around humans
    max_confidence = -1
    max_detection = None

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if classes[class_id] == "person":
                if confidence > max_confidence:
                    max_confidence = confidence
                    max_detection = detection

    detection = max_detection

    
    center_x = int(detection[0] * image.shape[1])
    center_y = int(detection[1] * image.shape[0])
    width = int(detection[2] * image.shape[1])
    height = int(detection[3] * image.shape[0])

    x = int(center_x - width/2)
    y = int(center_y - height/2)

    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + width > image.shape[0]:
        width = image.shape[0] - x
    if y + height > image.shape[1]:
        height = image.shape[1] - y

    #image = image[y:y+height, x:x+width]

    masks = mask_generator.generate(image)

    masks = sorted(masks, key=(lambda x: x['area']), reverse=True)

    for mask in masks:
        indices = np.where(mask['segmentation'].astype(np.uint8) != 0)
        points = list(zip(indices[1], indices[0]))
        epsilon = 0.02
        approx = cv2.approxPolyDP(np.array(points), epsilon, True)

        

        contour = cv2.convexHull(approx)

        M = cv2.moments(contour)
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])

        if x <= center_x <= x+width and y <= center_y <= y+height and mask['segmentation'][center_y,center_x]:
            largest_mask = mask
            break

    indices = np.where(largest_mask['segmentation'].astype(np.uint8) != 0)
    points = list(zip(indices[1], indices[0]))
    epsilon = 0.02
    approx = cv2.approxPolyDP(np.array(points), epsilon, True)

    i = 0
    print(len(masks))

    while i < (len(masks)):

        center = masks[i]['point_coords'][0]
        center = [int(x) for x in center]
        
        if cv2.pointPolygonTest(approx, center, measureDist=False) >= 0 and masks[i]['area'] > 50000:

            mask = Image.fromarray(masks[i]['segmentation'])
            mask = mask.convert('L')

            # Convert the mask to a boolean array
            mask_array = np.array(mask)
            mask_bool = mask_array.astype(bool)

            # Apply the mask to the image
            cropped_image = image.copy()
            cropped_image = cropped_image * np.expand_dims(mask_bool, axis=2)

            # Save the cropped image
            cropped_image = Image.fromarray(cropped_image)
            cropped_image.save(os.path.join('cropped', str(i)+"_"+file_name))

            i += 1

            
        else:
            masks.pop(i)