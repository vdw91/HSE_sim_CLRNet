import os
import cv2
import argparse
import numpy as np
import random
# from tqdm import tqdm

# initialize yolo model
from ultralytics import YOLO
model = YOLO('yolov8l.pt')  # load an official model


# Video information
video_path = 'CamID_59_20230712_152820230712_152843.mkv'
# video_path = 'downtown-rain-0003.mov'
# video_path = 'highway-night-0005.mov'
# row, col, channel = 1080, 1920, 3
row, col, channel = 720, 1280, 3
frame_rate = 30


# initialising background and foreground
BACKGROUND = np.zeros([row,col],np.uint8)
FOREGROUND = np.zeros([row,col],np.uint8)
a = np.uint8([255])
b = np.uint8([0])
noise_remove_kernel = np.ones([3,3],np.uint8)


def first_stage_bg_subtraction(frame):

    global BACKGROUND, FOREGROUND, a, b , noise_remove_kernel

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

    return BACKGROUND, FOREGROUND



def main():
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()

    row, col, channel = frame.shape

    print(row, col, channel)

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
            cv2.imshow('Frame',frame)


            first_stage_bg_subtraction(frame)

            yolo_seg_results = model(frame)  # predict on an image
            yolo_seg_results_plotted = yolo_seg_results[0].plot()

            # Process Yolo results list
            for result in yolo_seg_results:
                boxes = result.boxes  # Boxes object for bbox outputs
                masks = result.masks  # Masks object for segmentation masks outputs
                keypoints = result.keypoints  # Keypoints object for pose outputs
                probs = result.probs  # Class probabilities for classification outputs
                print(masks)
            
            
            cv2.imshow('background',BACKGROUND)
            cv2.imshow('foreground',FOREGROUND)
            cv2.imshow('yolo_seg_results_plotted',yolo_seg_results_plotted)
            
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.imwrite('frame_' + str(frame_count).zfill(6) + '.png', frame)
                cv2.imwrite('background.png', BACKGROUND)
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

if __name__ == '__main__':
    main()
