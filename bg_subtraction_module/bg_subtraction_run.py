import os
import cv2
import argparse
import numpy as np
import random
# from tqdm import tqdm

video_path = 'downtown-rain-0003.mov'



def main():
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()

    row, col, channel = frame.shape

    print(row, col, channel)

    # initialising background and foreground
    background = np.zeros([row,col],np.uint8)
    foreground = np.zeros([row,col],np.uint8)

    # converting data type of intergers 0 and 255 to uint8 type
    a = np.uint8([255])
    b = np.uint8([0])

    # creating kernel for removing noise
    kernel = np.ones([3,3],np.uint8)


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


            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
            # applying algorithm of median approximation method to get estimated background
            background = np.where(gray_frame>background,background+1,background-1)

            # using cv2.absdiff instead of background - frame, because 1 - 2 will give 255 which is not expected
            foreground = cv2.absdiff(background,gray_frame)
            
            # setting a threshold value for removing noise and getting foreground
            foreground = np.where(foreground>40,a,b)
            
            # removing noise
            foreground = cv2.erode(foreground,kernel)
            foreground = cv2.dilate(foreground,kernel)
            # using bitwise and to get colored foreground
            foreground = cv2.bitwise_and(frame, frame, mask=foreground)
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
