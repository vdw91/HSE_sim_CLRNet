import os
import cv2
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
import json
import labelme
import base64
# from tqdm import tqdm

# initialize yolo model
from ultralytics import YOLO
yolov8_detection_model = YOLO('yolov8l.pt')  # load an official model

# Video information
# video_path = 'CamID_59_20230713_102533_05.mkv'
video_path = 'CamID_04_Normal_20230727_121236.mkv'
# video_path = 'CamID_59_20230713_102533_02.mkv'
# video_path = 'CamID_46_Normal_20230717_110508.mkv'
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
lane_roi_kernel = np.ones([7,7],np.uint8)

def generate_photo_for_label():
    vid_list = [
        'CamID_59_Normal_20230727_190239.mkv',
        'CamID_59_20230713_102533_02.mkv',
        'CamID_04_Normal_20230727_121236.mkv',
        'CamID_46_Normal_20230717_110508.mkv',
        'CamID_62_Normal_20230717_144642_12.mkv'
    ]
    
    # convert yolov8 results to a list of shape dict (labelme format)
    def cvt_yolov8_result_to_a_shape_list(yolov8_ret_data):
        shape_list = []
        for an_obj in yolov8_ret_data:
            if(an_obj[5] == 2 or an_obj[5] == 3 or an_obj[5] == 5 or an_obj[5] == 7):
                shape_dict = {
                    "label": "car",
                    "points": [
                        [
                        191.0,
                        107.36900369003689
                        ],
                        [
                        313.0,
                        329.36900369003695
                        ]
                    ],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                }
                if(an_obj[5] == 2):
                    shape_dict['label'] = "car"

                if(an_obj[5] == 3):
                    shape_dict['label'] = "motorcycle"

                if(an_obj[5] == 5):
                    shape_dict['label'] = "bus"

                if(an_obj[5] == 7):
                    shape_dict['label'] = "truck"

                shape_dict['points'][0] = [float(an_obj[0]), float(an_obj[1])]
                shape_dict['points'][1] = [float(an_obj[2]), float(an_obj[3])]
                
                shape_list.append(shape_dict)

            else:
                continue

        return shape_list

    def run_label_generating(vid_path):

        vid_name = vid_path.split('.')[0]
        result_p = Path('datas/' + vid_name)
        result_p_0 = Path('datas/' + vid_name + '/images')
        result_p_1 = Path('datas/' + vid_name + '/labels')
        if(result_p.exists()):
            print('Save folder is created!')
            print(str(result_p))
        else:
            os.mkdir(result_p)
            os.mkdir(result_p_0)
            os.mkdir(result_p_1)
            print('Create ' + str(result_p))
            print('Create ' + str(result_p_0))
            print('Create ' + str(result_p_1))

        print(vid_path)
        cap = cv2.VideoCapture(vid_path)
        _, frame = cap.read()

        row, col, channel = frame.shape
        frame_count = 0


        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")

        print(yolov8_detection_model)
        
        # Read until video is completed
        while(cap.isOpened()):
        # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                if(frame_count % 60 == 0 and frame_count > 0):
                    
                    save_img_pth = str(result_p_0) + '/' + vid_name + '_' + str(frame_count).zfill(6) + '.png'
                    cv2.imwrite(save_img_pth, frame)
                    yolo_detection_results = yolov8_detection_model(frame)  # predict on an image

                    yolov8_ret_data = yolo_detection_results[0].cpu().boxes.data.numpy()
                    detection_plotted_img = draw_tracking_results(frame, yolov8_ret_data)

                    label_me_frame_dict = {
                        "version": "5.2.1",
                        "flags": {},
                        "shapes": [],
                        "imagePath": "2011_000003.jpg",
                        "imageData": None,
                        "imageHeight": 338,
                        "imageWidth": 500
                    }
                    data = labelme.LabelFile.load_image_file(save_img_pth)
                    image_data = base64.b64encode(data).decode('utf-8')

                    label_me_frame_dict['imageData'] = image_data
                    label_me_frame_dict['imagePath'] = vid_name + '_' + str(frame_count).zfill(6) + '.png'
                    label_me_frame_dict['imageHeight'] = col
                    label_me_frame_dict['imageWidth'] = row
                    shape_list = cvt_yolov8_result_to_a_shape_list(yolov8_ret_data)
                    # print(shape_list)
                    label_me_frame_dict['shapes'] = shape_list
                    json_file = json.dumps(label_me_frame_dict)

                    # print(label_me_frame_dict)
                    
                    
                    with open(str(result_p_1) + '/' + vid_name + '_' + str(frame_count).zfill(6) + '.json', "w") as outfile:
                        outfile.write(json_file)


                    cv2.imshow('yolo_detect', detection_plotted_img)
                    cv2.waitKey(1)

                frame_count += 1
            
            # Break the loop
            else: 
                break

        # When everything done, release the video capture object
        cap.release()
    
        # Closes all the frames
        cv2.destroyAllWindows()

        return

    for a_vid_path in vid_list:
        run_label_generating(a_vid_path)


def main():
    # make_vid()
    generate_photo_for_label()
    
if __name__ == '__main__':
    main()
