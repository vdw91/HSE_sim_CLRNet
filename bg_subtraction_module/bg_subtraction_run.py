import os
import cv2
import argparse
import numpy as np
import random
# from tqdm import tqdm

from ultralytics import YOLO


# Video information
video_path = 'downtown-rain-0003.mov'
# video_path = 'highway-night-0005.mov'
row, col, channel = 1280, 1920, 3
frame_rate = 30


# initialising background and foreground
background = np.zeros([row,col],np.uint8)
foreground = np.zeros([row,col],np.uint8)
a = np.uint8([255])
b = np.uint8([0])
noise_remove_kernel = np.ones([3,3],np.uint8)


def first_stage_bg_subtraction(frame):

    # Convert frame to gray (1 channel)
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Apply algorithm of median approximation method to get estimated background
    background = np.where(gray_frame>background,background+1,background-1)

    # Use cv2.absdiff instead of background - frame, because 1 - 2 will give 255 which is not expected
    foreground = cv2.absdiff(background,gray_frame)

    # setting a threshold value for removing noise and getting foreground
    foreground = np.where(foreground>40,a,b)
            
    # removing noise
    foreground = cv2.erode(foreground,noise_remove_kernel)
    foreground = cv2.dilate(foreground,noise_remove_kernel)
    # using bitwise and to get colored foreground
    foreground = cv2.bitwise_and(frame, frame, mask=foreground)



def main():
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()

    row, col, channel = frame.shape

    print(row, col, channel)


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
            
            
            cv2.imshow('background',background)
            cv2.imshow('foreground',foreground)
        
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.imwrite('frame.png', frame)
                cv2.imwrite('background.png', background)
                break
        
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
