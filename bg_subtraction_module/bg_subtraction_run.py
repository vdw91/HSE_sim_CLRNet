import os
import cv2
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
# from tqdm import tqdm

# initialize yolo model
from ultralytics import YOLO
yolov8_detection_model = YOLO('yolov8l.pt')  # load an official model
# from ultralytics import SAM
# yolov8_SAM_model = SAM('sam_l.pt')


# Video information
video_path = 'CamID_59_20230713_102533_05.mkv'
# video_path = 'downtown-rain-0003.mov'
# video_path = 'highway-night-0005.mov'
# row, col, channel = 1080, 1920, 3
row, col, channel = 720, 1280, 3
padding = 100
frame_rate = 30


# Segment Anything Module
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
device = "cuda"
sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=128,
    pred_iou_thresh=0.5,
    stability_score_thresh=0.9,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=300,  # Requires open-cv to run post-processing
)
SEGMENT_VIS = np.zeros([row,col, 3],np.uint8)


# initialising background and foreground
BACKGROUND = np.zeros([row,col],np.uint8)
BACKGROUND_COLORED = np.zeros([row,col, 3],np.uint8)
FOREGROUND = np.zeros([row,col],np.uint8)
FOREGROUND_COLORED = np.zeros([row,col, 3],np.uint8)
a = np.uint8([255])
b = np.uint8([0])
noise_remove_kernel = np.ones([3,3],np.uint8)

def fit_all_to_a_FULLHD(background, foreground, frame, detection, segment_res):
    global row, col, channel, padding
    vis_res = np.zeros((row * 2 + padding, col * 3 + padding * 2, channel))

    # First row - Frame - Background - Foreground
    vis_res[0:row, 0:col, : ] = frame
    vis_res[0:row, col + padding : col * 2 + padding, : ] = background
    vis_res[0:row, col * 2 + padding * 2 : , : ] = foreground

    # Second row - detection - segment_res - ***
    vis_res[row + padding : , 0:col, : ] = detection
    vis_res[row + padding : , col + padding : col * 2 + padding, : ] = segment_res

    return vis_res.astype(np.uint8)

def generate_mask(anns, default_frame):

    segment_result = default_frame.copy()

    _opacity_val = 0.35

    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3), dtype=segment_result.dtype)

    for ann in sorted_anns:
        m = ann['segmentation']
        random_color = (np.random.random(3) * 255 ).astype(np.uint8)

        img[m] = random_color

        segment_result[m] = cv2.addWeighted(segment_result[m], 1 - _opacity_val, img[m], _opacity_val, 1.0)

    return segment_result
    

def first_stage_bg_subtraction(frame):

    global BACKGROUND, BACKGROUND_COLORED, FOREGROUND, FOREGROUND_COLORED, a, b , noise_remove_kernel

    # Convert frame to gray (1 channel)
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Apply algorithm of median approximation method to get estimated background
    BACKGROUND = np.where(gray_frame>BACKGROUND,BACKGROUND+1,BACKGROUND-1)

    # Use cv2.absdiff instead of background - frame, because 1 - 2 will give 255 which is not expected
    FOREGROUND = cv2.absdiff(BACKGROUND,gray_frame)

    # setting a threshold value for removing noise and getting foreground
    FOREGROUND = np.where(FOREGROUND>40,a,b)
            
    # removing noise
    FOREGROUND = cv2.erode(FOREGROUND,noise_remove_kernel)
    FOREGROUND = cv2.dilate(FOREGROUND,noise_remove_kernel)
    # using bitwise and to get colored foreground
    FOREGROUND = cv2.bitwise_and(frame, frame, mask=FOREGROUND)

    BACKGROUND_COLORED = cv2.cvtColor(BACKGROUND,cv2.COLOR_GRAY2BGR)
    #FOREGROUND_COLORED = cv2.cvtColor(FOREGROUND,cv2.COLOR_GRAY2BGR)

    #return BACKGROUND, FOREGROUND, BACKGROUND_COLORED


def main():
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()

    row, col, channel = frame.shape
    frame_count = 0


    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
 
    # Read until video is completed
    while(cap.isOpened()):
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
        
            # Display the resulting frame
            # cv2.imshow('Frame',frame)

            vis_frame = frame.copy()

            first_stage_bg_subtraction(frame)
            
            yolo_seg_results = yolov8_detection_model(frame)  # predict on an image
            yolo_detection_plotted = yolo_seg_results[0].plot()

            if(frame_count > 0 and frame_count % 300 == 0):
                global SEGMENT_VIS
                masks = mask_generator.generate(BACKGROUND_COLORED)
                SEGMENT_VIS = generate_mask(masks, BACKGROUND_COLORED)
                # cv2.imwrite('segment_result_' + str(frame_count).zfill(6) + '.png', segment_result)

                # plt.axis('off')
                # plt.show() 
                #break

            # Process Yolo results list
            # for result in yolo_seg_results:
            #     boxes = result.boxes  # Boxes object for bbox outputs
            #     masks = result.masks  # Masks object for segmentation masks outputs
            #     keypoints = result.keypoints  # Keypoints object for pose outputs
            #     probs = result.probs  # Class probabilities for classification outputs
            #     print(masks)
            
            
            vis_res = fit_all_to_a_FULLHD(BACKGROUND_COLORED, FOREGROUND, frame, yolo_detection_plotted, SEGMENT_VIS)
            vis_res = cv2.resize(vis_res, [int(vis_res.shape[1] / 3), int(vis_res.shape[0] / 2)])

            cv2.imshow('vis', vis_res)
            # cv2.imshow('background',BACKGROUND_COLORED)
            # cv2.imshow('foreground',FOREGROUND)
            # cv2.imshow('yolo_seg_results_plotted',yolo_seg_results_plotted)
            
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                # cv2.imwrite('frame_' + str(frame_count).zfill(6) + '.png', frame)
                # cv2.imwrite('background.png', BACKGROUND)
                break

            frame_count += 1
        
        # Break the loop
        else: 
            break
    
    # When everything done, release the video capture object
    cap.release()
 
    # Closes all the frames
    cv2.destroyAllWindows()

    return 

def test_segment_anything():
    image = cv2.imread('Capture1.JPG')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    segment_result = generate_mask(masks, image)
    cv2.imwrite('segment_result.png', segment_result)

if __name__ == '__main__':
    # test_segment_anything()
    main()
