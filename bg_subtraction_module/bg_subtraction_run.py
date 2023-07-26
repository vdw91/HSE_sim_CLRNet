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


# Include tracker
from boxmot import DeepOCSORT
from pathlib import Path
tracker = DeepOCSORT(
  model_weights=Path('osnet_x0_25_msmt17.pt'),  # which ReID model to use
  device='cuda:0',  # 'cpu', 'cuda:0', 'cuda:1', ... 'cuda:N'
  fp16=True,  # wether to run the ReID model with half precision or not
)

# Video information
# video_path = 'CamID_59_20230713_102533_05.mkv'
video_path = 'CamID_46_Normal_20230717_110508.mkv'
row, col, channel = 720, 1280, 3

# video_path = 'highway-night-0005.mov'
# video_path = 'downtown-rain-0003.mov'
# row, col, channel = 1080, 1920, 3

padding = 100
frame_rate = 30


# Segment Anything Module
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
device = "cuda"
sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.5,
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


# Lane Detection configs and initialization
LANE_REGION_MASKS = np.zeros([row,col],np.uint8)
LANE_REGION_MASKS_RGB = np.zeros([row,col],np.uint8)

def update_lane_reg_mask():
    global LANE_REGION_MASKS, LANE_REGION_MASKS_RGB, FOREGROUND
    
    print(FOREGROUND.shape)
    # foreground_single_channel = cv2.cvtColor(FOREGROUND, cv2.COLOR_BGR2GRAY)

    LANE_REGION_MASKS = np.where(FOREGROUND == 255,LANE_REGION_MASKS+1,LANE_REGION_MASKS)
    LANE_REGION_MASKS_RGB = np.where(LANE_REGION_MASKS > 30 , 255, 0)
    print(LANE_REGION_MASKS_RGB.shape)


def fit_all_to_a_FULLHD(background, foreground, frame, detection, segment_res, lane_detection):
    global row, col, channel, padding
    vis_res = np.zeros((row * 2 + padding, col * 3 + padding * 2, channel))

    # First row - Frame - Background - Foreground
    vis_res[0:row, 0:col, : ] = frame
    vis_res[0:row, col + padding : col * 2 + padding, : ] = background
    vis_res[0:row, col * 2 + padding * 2 : , : ] = foreground

    # Second row - detection - segment_res - ***
    vis_res[row + padding : , 0:col, : ] = detection
    vis_res[row + padding : , col + padding : col * 2 + padding, : ] = segment_res
    vis_res[row + padding : , col * 2 + padding * 2 : , : ] = lane_detection

    return vis_res.astype(np.uint8)


def draw_tracking_results(im, tracking_results):

    draw_im = im.copy()
    for a_res in tracking_results:
        ori_x = int( (a_res[0] + a_res[2]) / 2 )
        ori_y = int( (a_res[1] + a_res[3]) / 2 )
        cv2.putText(draw_im, str(a_res[4]), (ori_x, ori_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (125,0,125), 3,  cv2.LINE_AA)

    return draw_im


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


def improved_bg_subtraction_using_Yolov8_Detection(frame, yolov8_results, frame_count):

    global BACKGROUND, BACKGROUND_COLORED, FOREGROUND, FOREGROUND_COLORED, a, b , noise_remove_kernel
    print('frame_count', frame_count)


    # Generate a mask which contain all vehicle detection from YoloV8
    def generate_yolov8_vehicle_mask(yolov8_results):
        yolov8_vehicle_mask = np.zeros(BACKGROUND.shape, dtype=np.uint8)

        for result in yolov8_results:
            boxes = result.boxes  # Boxes object for bbox outputs
            # masks = result.masks  # Masks object for segmentation masks outputs
            # keypoints = result.keypoints  # Keypoints object for pose outputs
            # probs = result.probs  # Class probabilities for classification outputs
            boxes_xyxy = boxes.xyxy
            boxes_cls = boxes.cls
            for a_box, a_cls in zip(boxes_xyxy, boxes_cls):
                #print(a_box, a_cls)
                if(a_cls == 2 or a_cls == 5 or a_cls == 7):
                    x_tl = int(a_box[0]) 
                    y_tl = int(a_box[1]) 
                    x_br = int(a_box[2]) 
                    y_br = int(a_box[3] * 1.05) 
                    yolov8_vehicle_mask[y_tl : y_br, x_tl : x_br] = 1

        # yolov8_vehicle_mask = np.where(BACKGROUND > 120, 1, yolov8_vehicle_mask)

        return yolov8_vehicle_mask
    

    def tracking_for_yolov8_detection(yolov8_results, im):
        # yolov8_results_cpu = yolov8_results.cpu()
        yolov8_ret_data = yolov8_results[0].cpu().boxes.data.numpy()
        # yolov8_ret_data_vehicle_only = yolov8_ret_data[yolov8_ret_data[:, 5] < ]
        tracker_outputs = tracker.update(yolov8_ret_data, im)
        return tracker_outputs


    # Apply yolov8_vehicle_mask for every 30 frames (which is the current framerate)
    # The period can vary based on the framerate
    # yolov8_vehicle_mask = np.zeros(BACKGROUND.shape, dtype=np.uint8)
    # if(frame_count % frame_rate == 0 or True):
    yolov8_vehicle_mask = generate_yolov8_vehicle_mask(yolov8_results)
    tracking_outputs = tracking_for_yolov8_detection(yolov8_results, frame)
    tracking_plotted_img = draw_tracking_results(frame, tracking_outputs)


    # Convert frame to gray (1 channel)
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Apply algorithm of median approximation method to get estimated background
    background_clone = BACKGROUND.copy()
    BACKGROUND = np.where(gray_frame>BACKGROUND,BACKGROUND+1,BACKGROUND-1)
    BACKGROUND = np.where(yolov8_vehicle_mask==1, background_clone, BACKGROUND)

    # Use cv2.absdiff instead of background - frame, because 1 - 2 will give 255 which is not expected
    FOREGROUND = cv2.absdiff(BACKGROUND,gray_frame)

    # setting a threshold value for removing noise and getting foreground
    FOREGROUND = np.where(FOREGROUND>40,a,b)
            
    # removing noise
    FOREGROUND = cv2.erode(FOREGROUND,noise_remove_kernel)
    FOREGROUND = cv2.dilate(FOREGROUND,noise_remove_kernel)
    # using bitwise and to get colored foreground
    FOREGROUND_COLORED = cv2.bitwise_and(frame, frame, mask=FOREGROUND)

    BACKGROUND_COLORED = cv2.cvtColor(BACKGROUND,cv2.COLOR_GRAY2BGR)

    return tracking_outputs, tracking_plotted_img

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

            
            yolo_detection_results = yolov8_detection_model(frame)  # predict on an image
            # yolo_detection_plotted = yolo_seg_results[0].plot()

            update_lane_reg_mask()

            # first_stage_bg_subtraction(frame)
            tracking_results, tracking_plotted_img = improved_bg_subtraction_using_Yolov8_Detection(
                frame, yolo_detection_results, frame_count)
            
            
            vis_res = fit_all_to_a_FULLHD(BACKGROUND_COLORED, FOREGROUND_COLORED, frame, 
                                          tracking_plotted_img, SEGMENT_VIS, SEGMENT_VIS)
            vis_res = cv2.resize(vis_res, [int(vis_res.shape[1] / 3), int(vis_res.shape[0] / 2)])


            cv2.imshow('vis', vis_res)
            lane_region = cv2.cvtColor(LANE_REGION_MASKS_RGB.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            cv2.imshow('lane_reg', lane_region)
            # cv2.imshow('lane_reg', LANE_REGION_MASKS_RGB)
            if(frame_count % 1000 == 0):
                # cv2.imwrite('results/' + str(frame_count).zfill(6) + '_vis.png', vis_res)
                masks = mask_generator.generate(BACKGROUND_COLORED)
                segment_result = generate_mask(masks, BACKGROUND_COLORED)
                # cv2.imwrite('results/' + str(frame_count).zfill(6) + '_seg.png', segment_result)
                # cv2.imwrite('results/' + str(frame_count).zfill(6) + '_bg.png', BACKGROUND_COLORED)
                

            # cv2.imshow('background',BACKGROUND_COLORED)
            # cv2.imshow('foreground',FOREGROUND)
            # cv2.imshow('yolo_seg_results_plotted',yolo_seg_results_plotted)
            
            # Press S on keyboard to do segment on background image
            if cv2.waitKey(73) & 0xFF == ord('s'):
                masks = mask_generator.generate(BACKGROUND_COLORED)
                segment_result = generate_mask(masks, BACKGROUND_COLORED)
                cv2.imshow("segment", segment_result)
                cv2.waitKey(0)
                #break


            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                # cv2.imwrite('frame_' + str(frame_count).zfill(6) + '.png', frame)
                # cv2.imwrite('vis.png', vis_res)
                
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
