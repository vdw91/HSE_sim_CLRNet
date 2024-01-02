# ==================================================================== #
# Copyright (C) 2023 - Automation Lab - Sungkyunkwan University
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
# ==================================================================== #

import os
import cv2
import argparse
import numpy as np
import random
from pathlib import Path

import base64
import math

from mmdet.apis import init_detector, inference_detector

from norfair import Detection, Tracker, Video, draw_tracked_objects, get_cutout
from norfair.filter import OptimizedKalmanFilterFactory

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from utils import Pita_Util

pita_Util_module = Pita_Util('')

from background_subtraction import Background_Subtraction_Module
from track_grid_module import Track_Grid_Module
from lane_divider import Lane_Divider_Module

# Global path setting
G_Weight_Path = 'weights/'
G_Save_Result_Path = 'results/'
G_Background_Path = 'results/bg_sub/'
G_Background_Stop_Frame = 15000
G_Grid_Path = 'results/grid/'
G_Grid_Stop_Frame = 15000
G_Lane_F = False
G_Save_step = 500


# NMS for RTM model
def _nms(RTMDet_results):
    import torchvision
    import torch
    # pred_dict = RTMDet_results['predictions']
    pred_dict = RTMDet_results
    labels = pred_dict['labels'].cpu().detach().numpy()
    scores = pred_dict['scores'].cpu().detach().numpy()
    bboxes = pred_dict['bboxes'].cpu().detach().numpy()

    nms_resutls = torchvision.ops.nms(torch.from_numpy(bboxes),
                                      torch.from_numpy(scores),
                                      iou_threshold=0.1)
    return labels, scores, bboxes, nms_resutls.cpu().detach().numpy()


def rtmdet_initialize():
    config_file = 'mmdetection/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py'
    checkpoint_file = G_Weight_Path + 'rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'

    detection_model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'

    return detection_model


def sam_initialize():
    device = "cuda:0"
    sam = sam_model_registry["default"](checkpoint=G_Weight_Path + "sam_vit_h_4b8939.pth")
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=128,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.9,
        # stability_score_offset = 1.0,
        # box_nms_thresh = 0.7,
        # crop_n_layers = 0,
        # crop_nms_thresh = 0.7,
        # crop_overlap_ratio = 512 / 1500,
        # crop_n_points_downscale_factor = 1,
        # point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area=250,
    )

    return mask_generator


def get_first_frame_from_vid(vid_path):
    cap = cv2.VideoCapture(vid_path)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    _, first_frame = cap.read()
    h, w, _ = first_frame.shape
    cap.release()

    return first_frame, h, w


def run_bg_subtraction(vid_path):
    detection_model = rtmdet_initialize()

    vid_name = vid_path.split('/')[-1].split('.')[0]
    bg_img_pth = G_Background_Path + vid_name
    if (Path.exists(Path(bg_img_pth)) == False):
        os.mkdir(bg_img_pth)

    first_frame, h, w = get_first_frame_from_vid(vid_path)
    bg_sub_module = Background_Subtraction_Module(first_frame)

    frame_counter = 0
    cap = cv2.VideoCapture(vid_path)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    _, frame = cap.read()

    stop_cond = False

    while (cap.isOpened()):
        print(vid_path, frame_counter)
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:

            # Get ROI only
            frame[600:, :, :] = 0

            RTMDet_results = inference_detector(detection_model, frame).pred_instances.to_dict()

            labels, scores, bboxes, nms_RTMDet = _nms(RTMDet_results)
            print('\nframe counter: ', frame_counter)
            print('\nDefault_RTM: ', len(RTMDet_results['labels']), ' NMS: ', len(nms_RTMDet), '\n')

            bg_sub_module.update_usingRTMDet(frame, labels, scores, bboxes, nms_RTMDet)

            # background_colored = bg_sub_module.background_v2
            background_colored = cv2.cvtColor(bg_sub_module.background, cv2.COLOR_GRAY2BGR)

            if (frame_counter % 500 == 0):
                cv2.imshow('background_colored', background_colored)
                save_pth = bg_img_pth + '/' + str(frame_counter).zfill(6) + '.png'
                cv2.imwrite(save_pth, background_colored)

            if (frame_counter == G_Background_Stop_Frame):
                stop_cond = True

            if (stop_cond == True):
                return background_colored, detection_model

            cv2.waitKey(1)
            frame_counter += 1
        else:
            # with open(results_path + '/_grid_vehicle_pixel.npy', 'wb') as f:
            #     np.save(f, rtmdet_module.grid_vehicle_pixel)
            # with open(results_path + '/_grid_vehicle_mask.npy', 'wb') as f:
            #     np.save(f, rtmdet_module.grid_vehicle_mask)
            break

    # When everything done, release the video capture object
    cap.release()


def remove_notmoving_obj(tracked_objs, moving_threshold=50):
    obj_status = np.ones((len(tracked_objs)))

    for idx in range(0, len(tracked_objs)):
        cur_obj = tracked_objs[idx]
        obj_past_detections = cur_obj.past_detections

        start_detection = obj_past_detections[0].points[0]
        end_detection = obj_past_detections[-1].points[0]

        dist_x = (start_detection[0] - end_detection[0])
        dist_y = (start_detection[1] - end_detection[1])
        dist = math.sqrt(dist_x ** 2 + dist_y ** 2)

        if (dist < moving_threshold):
            obj_status[idx] = 0

    return obj_status


def generate_grid_RoI(grid_active_points, bg_img):
    hulls = cv2.convexHull(grid_active_points)

    myROI = np.zeros((hulls.shape[0], 2), dtype=np.int32)

    for idx_hull in range(0, len(hulls)):
        myROI[idx_hull] = (hulls[idx_hull][0][0], hulls[idx_hull][0][1])

    mask = np.zeros((bg_img.shape[0], bg_img.shape[1]))
    mask = cv2.fillPoly(mask, [np.array(myROI)], 1)

    return mask


def segment_bg_wSAM(bg_img, mask_generator):
    def generate_keep_mask_list(anns, bg_img_white_mask):
        keep_id_list = []

        if len(anns) == 0:
            return keep_id_list
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

        bg_img_white_mask = bg_img_white_mask / 255

        img_all = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
        img_filtered = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
        # img[:,:,3] = 0
        for idx_m in range(0, len(sorted_anns)):
            m = sorted_anns[idx_m]['segmentation']
            m_area = np.sum(m)
            m_white_area = np.sum(m * bg_img_white_mask)

            color_mask = np.random.random(3) * 255
            img_all[m] = color_mask
            if (m_area < 500 and m_area > 50):
                if (m_white_area / m_area > 0.25):
                    keep_id_list.append(idx_m)
                    img_filtered[m] = color_mask

        # cv2.imshow('SAM results', img)
        # cv2.imshow('SAM results remove', img_test)

        return sorted_anns, keep_id_list, img_all, img_filtered

    sam_masks = mask_generator.generate(bg_img.copy())

    _, bg_img_white_mask = cv2.threshold(cv2.cvtColor(bg_img.copy(), cv2.COLOR_BGR2GRAY), 150, 255, cv2.THRESH_BINARY)

    sorted_masks, keep_mask_list, sam_normal_img, sam_filtered_img = generate_keep_mask_list(sam_masks,
                                                                                             bg_img_white_mask)

    return sorted_masks, keep_mask_list


def generate_lane_lines(sam_masks, keep_mask_list, grid_ap, grid_mask, bg_img):
    def display_vanishing_point(frame, vanishing_point):
        cv2.circle(frame, vanishing_point, 3, (0, 255, 0), 3, cv2.LINE_AA)
        return frame

    def get_inliers_w_vanishing_point(vp, lines, threshold_dist2vp=10):
        inliers = []

        for line in lines:
            dist = find_dist_to_line(vp, line)
            if (dist < threshold_dist2vp):
                inliers.append(line)

        return inliers

    def find_dist_to_line(point, line):
        """Implementation is based on Computer Vision material, owned by the University of Melbourne
        Find an intercept point of the line model with a normal from point to it, to calculate the distance betwee point and intercept
        Args: point: the point using x and y to represent
        line: the line using rho and theta (polar coordinates) to represent
        Return: dist: the distance from the point to the line
        """
        x0, y0 = point
        rho, theta = line[0]
        m = (-1 * (np.cos(theta))) / np.sin(theta)
        c = rho / np.sin(theta)
        # intersection point with the model
        x = (x0 + m * y0 - m * c) / (1 + m ** 2)
        y = (m * x0 + (m ** 2) * y0 - (m ** 2) * c) / (1 + m ** 2) + c
        dist = math.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        return dist

    def find_intersection_point(line1, line2):
        """Implementation is based on code from https://stackoverflow.com/questions/46565975, Original author: StackOverflow contributor alkasm
        Find an intercept point of 2 lines model
        Args: line1,line2: 2 lines using rho and theta (polar coordinates) to represent
        Return: x0,y0: x and y for the intersection point
        """
        # rho and theta for each line
        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]
        # Use formula from https://stackoverflow.com/a/383527/5087436 to solve for intersection between 2 lines
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        det_A = np.linalg.det(A)
        if det_A != 0:
            x0, y0 = np.linalg.solve(A, b)
            # Round up x and y because pixel cannot have float number
            x0, y0 = int(np.round(x0)), int(np.round(y0))
            return x0, y0
        else:
            return None

    def RANSAC(lines, ransac_iterations, ransac_threshold, ransac_ratio, ap_top):
        """Implementation is based on code from Computer Vision material, owned by the University of Melbourne
        Use RANSAC to identify the vanishing points for a given image
        Args: lines: The lines for the image
        ransac_iterations,ransac_threshold,ransac_ratio: RANSAC hyperparameters
        Return: vanishing_point: Estimated vanishing point for the image
        """
        inlier_count_ratio = 0.
        vanishing_point = (0, 0)
        # perform RANSAC iterations for each set of lines
        for iteration in range(ransac_iterations):
            # randomly sample 2 lines
            n = 2
            selected_lines = random.sample(lines, n)
            line1 = selected_lines[0]
            line2 = selected_lines[1]
            intersection_point = find_intersection_point(line1, line2)

            if intersection_point is not None:
                if (intersection_point[1] <= ap_top):
                    # count the number of inliers num
                    inlier_count = 0
                    # inliers are lines whose distance to the point is less than ransac_threshold
                    for line in lines:
                        # find the distance from the line to the point
                        dist = find_dist_to_line(intersection_point, line)
                        # check whether it's an inlier or not
                        if dist < ransac_threshold:
                            inlier_count += 1

                    # If the value of inlier_count is higher than previously saved value, save it, and save the current point
                    if inlier_count / float(len(lines)) > inlier_count_ratio:
                        inlier_count_ratio = inlier_count / float(len(lines))
                        vanishing_point = intersection_point

                    # We are done in case we have enough inliers
                    if inlier_count > len(lines) * ransac_ratio:
                        break
        return vanishing_point

    def get_top_ap(grid_ap):
        top_x = 2480
        for ap in grid_ap:
            if (ap[1] < top_x):
                top_x = ap[1]

        return top_x

    def get_valid_lines(lines):
        valid_lines = []
        # Remove horizontal and vertical lines as they would not converge to vanishing point
        for line in lines:
            rho, theta = line[0]
            valid_lines.append(line)

        return valid_lines

    def display_lines_Rho_Theta(frame, lines):
        if lines is not None:
            for line in lines:
                rho = line[0][0]
                theta = line[0][1]

                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

                cv2.line(frame, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

                # print('rho, theta: ', rho, theta)
        return frame

    def generate_sam_mask(sam_masks, keep_mask_list, grid_mask, bg_img):
        final_sam_mask = np.zeros((bg_img.shape[0], bg_img.shape[1]))

        for idx_m in keep_mask_list:
            m = sam_masks[idx_m]['segmentation']
            m_area = sam_masks[idx_m]['area']
            value_grid = np.sum(m * grid_mask)

            if (value_grid > 0 and m_area > 20):
                final_sam_mask[m] = 1

        final_sam_img = (final_sam_mask * 255).astype(np.uint8)
        return final_sam_img

    def upgrade_sam_mask(sam_masks, keep_mask_list, grid_mask, bg_img):

        sam_mask = generate_sam_mask(sam_masks, keep_mask_list, grid_mask, bg_img)

        contours, hierarchy = cv2.findContours(sam_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        selected_contours = [c for c in contours if (
                (
                        cv2.minAreaRect(c)[1][0] < 8
                        or
                        cv2.minAreaRect(c)[1][1] < 8
                )
                and
                (
                        cv2.minAreaRect(c)[2] < 85
                        and
                        cv2.minAreaRect(c)[2] > 10
                )
        )]

        draw_mat = np.zeros(bg_img.shape)
        s_cnt_img = cv2.drawContours(draw_mat, selected_contours, -1, (255, 255, 255), 3)

        return s_cnt_img

    final_sam_img = upgrade_sam_mask(sam_masks, keep_mask_list, grid_mask, bg_img)
    # cv2.imshow('final_sam_img', final_sam_img)

    final_sam_img_gray = cv2.cvtColor(final_sam_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    _, final_sam_img_mask = cv2.threshold(final_sam_img_gray, 150, 255, cv2.THRESH_BINARY)
    mask_white_mask_canny = cv2.Canny(final_sam_img_mask, 1, 10, None, 3)
    # cv2.imshow('Canny', mask_white_mask_canny)

    lines_RT = cv2.HoughLines(mask_white_mask_canny, 1, np.pi / 360, 50, None, 0, 0)
    if (lines_RT is not None):
        # grouped_lines, means_lines = group_lines_by_rho(lines_RT)
        # show_lines_by_group(grouped_lines, bg_img.copy())
        valid_lines = get_valid_lines(lines_RT)

        print('Number of lines: ', len(valid_lines))
        # hough_RT_visual = display_lines_Rho_Theta(bg_img.copy(), valid_lines)

        # RANSAC parameters:
        ransac_iterations, ransac_threshold, ransac_ratio = 350, 10, 0.90
        ap_top = get_top_ap(grid_ap)
        # vanishing_point = RANSAC_2(valid_lines, ransac_iterations, ransac_threshold, ransac_ratio, ap_top)
        vanishing_point = RANSAC(valid_lines, ransac_iterations, ransac_threshold, ransac_ratio, ap_top)
        confirmed_lines = get_inliers_w_vanishing_point(vanishing_point, lines_RT)
        confirmed_lines_visual = display_lines_Rho_Theta(bg_img.copy(), confirmed_lines)
        confirmed_lines_visual = display_vanishing_point(confirmed_lines_visual, vanishing_point)

        # cv2.imshow("Final lane", confirmed_lines_visual)
        return final_sam_img, confirmed_lines_visual
    else:
        print('Hough can not find any lines')
        return final_sam_img, final_sam_img


def join_visualization(detect_track_frame, background, grid_ap_shown, grid_region_frame):
    h, w, c = detect_track_frame.shape
    vid_img = np.zeros((h * 2, w * 2, c), dtype=np.uint8)

    vid_img[0:h, 0: w, :] = detect_track_frame.copy()
    vid_img[0:h, w: w * 2, :] = background.copy()

    vid_img[h:h * 2, 0:w, :] = grid_ap_shown.copy()

    grid_region_frame_show = (grid_region_frame.copy() * 255).astype(np.uint8)
    grid_region_frame_show = cv2.cvtColor(grid_region_frame_show, cv2.COLOR_GRAY2BGR)
    vid_img[h:h * 2, w:w * 2, :] = grid_region_frame_show

    vid_img = cv2.resize(vid_img, (w, h))

    return vid_img


def run(vid_path):
    # Initialization
    detection_model = rtmdet_initialize()
    mask_generator = sam_initialize()
    global G_Background_Path, G_Background_Stop_Frame, G_Grid_Path, \
        G_Grid_Stop_Frame, G_Lane_F, G_Save_step, G_Save_Result_Path

    vid_name = vid_path.split('/')[-1].split('.')[0]

    # save_result_pth = G_Save_Result_Path + vid_name
    # if(Path.exists(Path(save_result_pth)) == False):
    #     os.mkdir(save_result_pth)

    bg_img_pth = G_Background_Path + vid_name
    if (Path.exists(Path(bg_img_pth)) == False):
        os.mkdir(bg_img_pth)

    grid_img_pth = G_Grid_Path + vid_name
    if (Path.exists(Path(grid_img_pth)) == False):
        os.mkdir(grid_img_pth)

    first_frame, h, w = get_first_frame_from_vid(vid_path)
    bg_sub_module = Background_Subtraction_Module(first_frame)
    track_grid_module = Track_Grid_Module(w, h)
    tracker = Tracker(
        initialization_delay=1,
        distance_function="sqeuclidean", distance_threshold=50, detection_threshold=0.3, past_detections_length=15
    )
    lane_divider_module = Lane_Divider_Module()

    frame_counter = 0
    cap = cv2.VideoCapture(vid_path)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    _, frame = cap.read()

    while (cap.isOpened()):
        print(vid_path, frame_counter)
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:

            # Get ROI only
            # frame[600:, :, :] = 0

            RTMDet_results = inference_detector(detection_model, frame).pred_instances.to_dict()

            labels, scores, bboxes, nms_RTMDet = _nms(RTMDet_results)
            print('\nframe counter: ', frame_counter)
            print('\nDefault_RTM: ', len(RTMDet_results['labels']), ' NMS: ', len(nms_RTMDet), '\n')
            print('\nActive grid point number: ', track_grid_module.list_of_active_grid().shape)

            # Background Subtraction
            bg_sub_module.update_usingRTMDet(frame, labels, scores, bboxes, nms_RTMDet)
            background_colored = cv2.cvtColor(bg_sub_module.background, cv2.COLOR_GRAY2BGR)

            # Grid Module
            ## Generate norfair tracking input
            nofair_detections = []
            if bboxes.shape[0] > 0:
                for idx_obj in nms_RTMDet:
                    # bbox is a 1 dimensional array with [x1, y1, x2, y2, score]
                    # if(labels[idx_obj] == 2 or labels[idx_obj] == 3 or labels[idx_obj] == 5 or labels[idx_obj] == 7):
                    # We exclude truck and bus from the tracking
                    if (labels[idx_obj] == 2 or labels[idx_obj] == 3):
                        centroid = bboxes[idx_obj][:4].reshape((2, 2)).mean(axis=0)
                        nofair_detections.append(
                            Detection(
                                centroid,
                                # bottom_centroid,
                                scores=np.array([scores[idx_obj]]),
                                label=labels[idx_obj],
                            )
                        )

            ## Update the track and remove the standstill objects
            tracked_objects = tracker.update(detections=nofair_detections)
            moving_obj_stt = remove_notmoving_obj(tracked_objects)
            for track_obj_idx in range(0, len(tracked_objects)):
                a_tracked_obj = tracked_objects[track_obj_idx]
                a_tracked_obj_moving_stt = moving_obj_stt[track_obj_idx]

                if (a_tracked_obj.live_points[0] == True and a_tracked_obj_moving_stt == 1):
                    track_grid_module.update_per_object(a_tracked_obj)

            draw_tracked_objects(frame, tracked_objects)

            grid_ap = track_grid_module.list_of_active_grid()
            grid_ap_show = track_grid_module.display_selected_point(grid_ap, background_colored.copy())
            grid_mask = np.zeros((h, w))
            if (len(grid_ap) > 10):
                grid_mask = generate_grid_RoI(grid_ap, background_colored.copy())

            # Visulization
            vis_img = join_visualization(frame, background_colored, grid_ap_show, grid_mask)
            cv2.imshow('ETS_Lane', vis_img)
            # cv2.imshow('Detection and Tracking', frame)
            # cv2.imshow('Grid', grid_mask)

            if (frame_counter % G_Save_step == 0 and frame_counter > 0):
                save_bg_pth = bg_img_pth + '/' + str(frame_counter).zfill(6) + '.png'
                cv2.imwrite(save_bg_pth, background_colored)
                save_grid_pth = grid_img_pth + '/' + str(frame_counter).zfill(6) + '.png'
                cv2.imwrite(save_grid_pth, grid_mask)

            if (frame_counter >= G_Background_Stop_Frame and frame_counter > G_Grid_Stop_Frame):
                G_Lane_F = True

            if (G_Lane_F):
                print('Calculate Lane')
                # SAM
                sam_masks, keep_mask_list = segment_bg_wSAM(background_colored.copy(), mask_generator)
                # Generate lane
                final_sam_img, final_lane = lane_divider_module.run(sam_masks, keep_mask_list,
                                                                    grid_ap, grid_mask, background_colored.copy())
                cv2.imshow('Lane', final_lane)
                # cv2.waitKey(1)

                # Save results
                save_bg_pth = bg_img_pth + '/' + str(frame_counter).zfill(6) + '_bg.png'
                save_grid_pth = grid_img_pth + '/' + str(frame_counter).zfill(6) + '_grid.png'
                save_lane_pth = G_Save_Result_Path + vid_name + '_' + str(frame_counter).zfill(6) + '_lane.png'
                cv2.imwrite(save_bg_pth, background_colored)
                cv2.imwrite(save_grid_pth, grid_mask)
                cv2.imwrite(save_lane_pth, final_lane)

                G_Lane_F = False
                break
            cv2.waitKey(1)
            frame_counter += 1
        else:
            break

    # When everything done, release the video capture object
    cap.release()


def main():
    dataset_path = 'dataset/selected_vid'

    vid_list_path = pita_Util_module.get_list_of_file_in_a_path(dataset_path)

    for vid_idx in range(0, len(vid_list_path)):
        vid_pth = dataset_path + '/' + vid_list_path[vid_idx]
        run(vid_pth)


if __name__ == '__main__':
    main()