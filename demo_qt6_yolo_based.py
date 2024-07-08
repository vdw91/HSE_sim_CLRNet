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


# sudo apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
# Or remove normal opencv and install opencv-headless
# https://github.com/NVlabs/instant-ngp/discussions/300

import os
import time

import cv2
import argparse
import numpy as np
import random
from pathlib import Path
import pickle

import base64
import math

# For demo QT6
# from PySide6 import QtCore, QtWidgets, QtGui, QMainWindow
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6 import uic

from utils import Pita_Util

pita_Util_module = Pita_Util('')

from lane_detection_yolo import Lane_Detection
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
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


class WorkerThread(QThread):
    update_progress = pyqtSignal(dict)
    update_lane_progress = pyqtSignal(dict)

    def __init__(self, vid_pth, run_all_flag=False):
        super().__init__()
        self.vid_path = vid_pth
        self.pause_flag = False
        self.stop_flag = False
        self.vid_list = None

        self.start_frame = [310, 180, 210, 270, 75, 110, 75, 550, 45, 150, 5, 25, 550, 5, 0, 90, 0, 30, 180, 70, 0]
        self.end_frame = [390, 260, 240, 300, 100, 170, 120, 600, 120, 210, 30, 85, 650, 90, 50, 165, 150, 105, 210,
                          165, 90]

        self.total_lanes = []
        self.detected_tp = []
        self.detected_fp = []
        self.detected_fn = []
        self.precision = []
        self.recall = []
        self.accuracy = []

        self.run_all_flag = run_all_flag

    def reset_flag_for_new_vid(self):
        self.pause_flag = False
        self.stop_flag = False

    def reset_flag_for_new_session(self):
        self.total_lanes = []
        self.detected_tp = []
        self.detected_fp = []
        self.detected_fn = []
        self.precision = []
        self.recall = []
        self.accuracy = []

    def get_first_frame_from_vid(self, vid_path):
        cap = cv2.VideoCapture(vid_path)
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        _, first_frame = cap.read()
        h, w, _ = first_frame.shape
        cap.release()

        return first_frame, h, w

    def segment_bg_wSAM(self, bg_img, mask_generator):

        def generate_keep_mask_list(anns, bg_img_white_mask):
            keep_id_list = []

            if len(anns) == 0:
                return keep_id_list
            sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

            bg_img_white_mask = bg_img_white_mask / 255

            img_all = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
            img_filtered = np.zeros(
                (sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
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

            return sorted_anns, keep_id_list, img_filtered

        def generate_sam_viz(anns, bg_img):
            keep_id_list = []

            if len(anns) == 0:
                return keep_id_list
            sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

            img_all = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4),
                               dtype=np.float32)
            img_all[:, :, 3] = 0
            img_all[:, :, 0:3] = bg_img

            for idx_m in range(0, len(sorted_anns)):
                m = sorted_anns[idx_m]['segmentation']
                color_mask = np.concatenate([np.random.random(3), [0.35]])
                img_all[m] = color_mask

            return cv2.cvtColor(img_all, cv2.COLOR_BGRA2BGR)

        sam_masks = mask_generator.generate(bg_img.copy())

        _, bg_img_white_mask = cv2.threshold(cv2.cvtColor(bg_img.copy(), cv2.COLOR_BGR2GRAY), 150, 255,
                                             cv2.THRESH_BINARY)

        sorted_masks, keep_mask_list, sam_normal_img = generate_keep_mask_list(sam_masks, bg_img_white_mask)
        sam_normal_img_all = generate_sam_viz(sam_masks, bg_img)

        return sorted_masks, keep_mask_list, sam_normal_img_all

    def evaluate_per_vid(self, vid_pth, h, w, vp, lane_point_list, bg_img):
        import json
        from shapely.geometry import Polygon

        def approximate_gt_to_line(points):
            no_points = len(points)

            f_p1 = 0
            f_p2 = 0
            min_dist = np.inf

            for p1_idx in range(0, no_points):
                for p2_idx in range(0, no_points):
                    cur_d = 0
                    for p3_idx in range(0, no_points):
                        if (p1_idx != p2_idx) and (p1_idx != p3_idx) and (p2_idx != p3_idx):
                            p1 = np.array([points[p1_idx][0], points[p1_idx][1]])
                            p2 = np.array([points[p2_idx][0], points[p2_idx][1]])
                            p3 = np.array([points[p3_idx][0], points[p3_idx][1]])
                            cur_d += np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
                    if cur_d < min_dist and cur_d != 0:
                        min_dist = cur_d
                        f_p1 = p1_idx
                        f_p2 = p2_idx

            # print(f_p1, f_p2)
            return points[f_p1], points[f_p2]

        def line_to_end_point(points, h, w):
            m = (points[0][1] - points[1][1]) / (points[0][0] - points[1][0])
            b = points[0][1] - m * points[0][0]

            x_low = ((h - 1) - b) / m
            x_high = - b / m

            return (int(x_high), 0), (int(x_low), h - 1,)

        def get_area_between_2_lines(detected_lane, gt_lane, line_width=30):

            def polygon_with_witdh(line, line_width):
                line_rect = Polygon([
                    [line[0][0] - line_width, line[0][1]],
                    [line[0][0] + line_width, line[0][1]],
                    [line[1][0] + line_width, line[1][1]],
                    [line[1][0] - line_width, line[1][1]]
                ]
                )
                return line_rect

            d_line_rect = polygon_with_witdh(detected_lane, line_width)
            g_line_rect = polygon_with_witdh(gt_lane, line_width)

            overlap_ratio = d_line_rect.intersection(g_line_rect).area / g_line_rect.area

            return overlap_ratio

        def get_gt_lane_point(gt_jsonf):
            lane_points = []
            f = open(gt_jsonf)
            data = json.load(f)
            f.close()

            shapes = data['shapes']
            for shape in shapes:
                points = shape['points']
                lp1, lp2 = approximate_gt_to_line(points)
                lane_points.append([lp1, lp2])

            return lane_points

        def reformat_detected_lanes(vp, lane_point_list, h):
            detected_lanes = []

            for line_p in lane_point_list:
                detected_lanes.append([int(vp[0]), int(vp[1]), int(line_p), h - 1])

            return detected_lanes

        gt_jsonf = vid_pth.replace('.mkv', '.json')
        gt_jsonf = gt_jsonf.replace('selected_vid', 'label')
        gt_lane = get_gt_lane_point(gt_jsonf)

        detected_lanes = reformat_detected_lanes(vp, lane_point_list, h)

        bg_mat = bg_img.copy()

        end_d_lanes = []
        end_g_lanes = []

        for lane in gt_lane:
            top_p, bot_p = line_to_end_point(lane, h, w)
            bg_mat = cv2.line(bg_mat, top_p, bot_p,
                              (0, 0, 255), 3, cv2.LINE_AA)
            end_g_lanes.append([top_p, bot_p])

        for d_lane in detected_lanes:
            top_p, bot_p = line_to_end_point([[d_lane[0], d_lane[1]], [d_lane[2], d_lane[3]]], h, w)
            bg_mat = cv2.line(bg_mat, top_p, bot_p,
                              (0, 255, 255), 3, cv2.LINE_AA)
            end_d_lanes.append([top_p, bot_p])

        area_lanes = np.full((len(end_d_lanes), 2), -1, dtype=np.float32)
        area_lanes_ratio = np.full((len(end_d_lanes), len(end_g_lanes)), 0, dtype=float)

        for idx_d, d_lane in enumerate(end_d_lanes):
            for idx_gt, gt_lane in enumerate(end_g_lanes):
                overlap_ratio = get_area_between_2_lines(d_lane, gt_lane)
                area_lanes_ratio[idx_d][idx_gt] = overlap_ratio

            best_ind_match = np.argmax(area_lanes_ratio[idx_d])
            area_lanes[idx_d][0] = best_ind_match
            area_lanes[idx_d][1] = area_lanes_ratio[idx_d][best_ind_match]

        final_tp = 0
        final_fp = 0
        final_fn = 0

        gt_matched_mat = np.zeros((len(end_g_lanes), 2))
        # print(area_lanes_ratio)
        # print(area_lanes)

        for area_lane in area_lanes:
            matched_gt_id, conf_score = area_lane
            matched_gt_id = int(matched_gt_id)
            if gt_matched_mat[matched_gt_id][1] == 0:
                gt_matched_mat[matched_gt_id][1] = conf_score
                gt_matched_mat[matched_gt_id][0] = matched_gt_id
            else:
                final_fp += 1
                if gt_matched_mat[matched_gt_id][1] < conf_score:
                    gt_matched_mat[matched_gt_id][1] = conf_score
                    gt_matched_mat[matched_gt_id][0] = matched_gt_id

        for final_lane_mat in gt_matched_mat:
            conf_score, lane_id = final_lane_mat

            if conf_score == 0:
                final_fn += 1
            else:
                final_tp += 1
        return bg_mat, final_tp, final_fp, final_fn

    def run_a_vid(self, cur_vid_pth, idx_vid):
        vid_name = cur_vid_pth.split('/')[-1].split('.')[0]
        print(vid_name)

        # bg_img_pth = G_Background_Path + vid_name
        # if (Path.exists(Path(bg_img_pth)) == False):
        #     os.mkdir(bg_img_pth)
        #
        # grid_img_pth = G_Grid_Path + vid_name
        # if (Path.exists(Path(grid_img_pth)) == False):
        #     os.mkdir(grid_img_pth)

        lane_cctv = Lane_Detection(show_flag=False)
        first_frame, h, w = self.get_first_frame_from_vid(cur_vid_pth)
        cap = cv2.VideoCapture(cur_vid_pth)
        lane_cctv.new_vid_input(first_frame, h, w)

        if (cap.isOpened() == False):
            print("Error opening video stream or file")

        frame_counter = 0

        while cap.isOpened() and self.stop_flag == False:
            # Capture frame-by-frame
            ret, im = cap.read()
            if self.stop_flag:
                break
            if ret:
                frame_log = ''
                if frame_counter >= 30 * self.start_frame[idx_vid]:
                    # Get ROI only
                    im[600:, :, :] = 0
                    det_track_ret, bg_ret, grid_mask_ret, grid_ap_ret, grid_mask = lane_cctv.run_debug(im)
                    frame_log = ''
                    frame_log += 'Frame counter: ' + str(frame_counter - 30 * self.start_frame[idx_vid])
                    frame_log += '\nActive grid point number:: ' + str(
                        lane_cctv.track_grid_module.list_of_active_grid().shape[0])

                    if frame_counter == 30 * self.end_frame[idx_vid] or self.pause_flag:
                        time.sleep(0)
                        sam_masks, keep_mask_list, sam_viz = self.segment_bg_wSAM(bg_ret.copy(),
                                                                                  lane_cctv.mask_generator)
                        lane_divider_module = Lane_Divider_Module()
                        final_sam_img, final_lane, vp, lane_p_list = lane_divider_module.run(sam_masks, keep_mask_list,
                                                                                             lane_cctv.track_grid_module.list_of_active_grid(),
                                                                                             grid_mask,
                                                                                             bg_ret.copy())
                        # save_dict = {
                        #     'bg_ret': bg_ret,
                        #     'sam_masks': sam_masks,
                        #     'keep_mask_list': keep_mask_list,
                        #     'sam_viz': sam_viz,
                        #     'list_of_active_grid': lane_cctv.track_grid_module.list_of_active_grid(),
                        #     'grid_mask': grid_mask
                        # }
                        #
                        # with open('results/temp_files/' + vid_name + '.pkl', 'wb') as save_dict_f:
                        #     pickle.dump(save_dict, save_dict_f)

                        eval_mat, final_tp, final_fp, final_fn = self.evaluate_per_vid(cur_vid_pth, h, w, vp, lane_p_list, bg_ret)
                        no_detected_lanes = final_tp + final_fp
                        final_P = 0
                        final_R = 0
                        final_f1 = 0
                        if final_tp > 0:
                            final_P = final_tp / (final_tp + final_fp)
                            final_R = final_tp / (final_tp + final_fn)
                            final_f1 = 2 * final_P * final_R / (final_P + final_R)


                        self.total_lanes.append(final_tp + final_fn)
                        self.detected_tp.append(final_tp)
                        self.detected_fp.append(final_fp)
                        self.detected_fn.append(final_fn)
                        self.precision.append(final_P)
                        self.recall.append(final_R)
                        self.accuracy.append(final_f1)

                        print(final_tp, final_fp, final_fn)

                        frame_log += '\nFinish'
                        self.stop_flag = True
                        self.pause_flag = False
                        self.update_lane_progress.emit({
                            'frame': im,
                            'vid_name': vid_name,
                            'background_colored': bg_ret,
                            'grid_ap_show': grid_ap_ret,
                            'grid_mask': grid_mask_ret,
                            # 'final_sam_img': final_sam_img,
                            'final_sam_img': eval_mat,
                            'final_lane_img': final_lane,
                            'no_detected_lanes': no_detected_lanes,
                            'final_tp': final_tp,
                            'final_fp': final_fp,
                            'final_fn': final_fn,
                            'final_f1': final_f1,
                            'final_P': final_P,
                            'final_R': final_R
                        })
                    else:
                        self.update_progress.emit({
                            'frame': im,
                            'background_colored': bg_ret,
                            'grid_ap_show': grid_ap_ret,
                            'grid_mask': grid_mask_ret,
                            'frame_log': frame_log
                        })

                else:
                    frame_log = 'Searching for a right frame to start'
                    # print(frame_log, frame_counter)
                frame_counter += 1


            else:
                break

        # When everything done, release the video capture object
        cap.release()

    def run(self):
        start_time = time.time()
        vid_name = self.vid_path.split('/')[-1]
        fol_pth = self.vid_path.replace(vid_name, '')
        vid_pth_list = pita_Util_module.get_list_of_file_in_a_path(fol_pth)
        self.vid_list = sorted(vid_pth_list)
        self.reset_flag_for_new_session()

        if self.run_all_flag:
            for vid_idx in range(0, len(self.vid_list)):
            # for vid_idx in range(0, 1):
                print(self.vid_path, self.vid_list[vid_idx])
                cur_vid_path = fol_pth + self.vid_list[vid_idx]
                self.run_a_vid(cur_vid_path, vid_idx)
                self.reset_flag_for_new_vid()
                time.sleep(5)

            # Write overall evaluation file
            save_result_pth = 'results/lane/'
            eval_file = open(save_result_pth + 'evaluation.txt', "w")
            eval_file.write("Total videos: %d\n"%len(self.vid_list))
            for vid_idx in range(0, len(self.vid_list)):
            # for vid_idx in range(0, 1):
                vid_name = self.vid_list[vid_idx].split('/')[-1].split('.')[0]
                eval_file.write('Video: %s\t' % vid_name +
                                'TP: %d\t' % self.detected_tp[vid_idx] +
                                'FP: %d\t' % self.detected_fp[vid_idx] +
                                'FN: %d\t' % self.detected_fn[vid_idx] +
                                'Precision: %f\t' % round(self.precision[vid_idx],2) +
                                'Recall: %f\t' % round(self.recall[vid_idx],2) +
                                'Acc: %f\n' % round(self.accuracy[vid_idx],2) )
            eval_file.write("\n--------------------------------------------------------\n" +
                            "Average results for all videos: \n" +
                            'Precision: %f\t' % round(np.average(self.precision), 2) +
                            'Recall: %f\t' % round(np.average(self.recall), 2) +
                            'Acc: %f\n' % round(np.average(self.accuracy), 2)
            )
            eval_file.close()

        else:
            for vid_idx in range(0, len(self.vid_list)):
                print(self.vid_path, self.vid_list[vid_idx])
                if vid_name == self.vid_list[vid_idx]:
                    self.run_a_vid(self.vid_path, vid_idx)
                    break
        end_time = time.time()
        print("Total running time: ", end_time - start_time)

class MyDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("cctv_lane_qt6.ui", self)
        self.setWindowTitle('CCTV Lane Detection Demo')

        self.select_vid_button.clicked.connect(self.browse_file)
        self.start_button.clicked.connect(self.start_click)
        self.estimate_save_button.clicked.connect(self.estimate_save_click)
        self.autorun_folder_button.clicked.connect(self.run_all_dataset)

        self.run_all_flag = False

    def cvt_num_rgb_img_to_qtpixmap(self, frame_img):
        rgb_img = frame_img.copy()
        rgb_img[:, :, 0] = frame_img[:, :, 2]
        rgb_img[:, :, 2] = frame_img[:, :, 0]
        h, w, _ = rgb_img.shape
        qimage = QImage(rgb_img.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap(qimage)
        pixmap = pixmap.scaled(400, 225)
        return pixmap

    def browse_file(self):
        fname = QFileDialog.getOpenFileName(self, '')
        vid_f_name = fname[0].split('/')[-1]
        self.vid_pth_text.setText(vid_f_name)
        self.cur_vid_pth = fname[0]

    def start_click(self):
        self.worker = WorkerThread(self.cur_vid_pth)
        self.worker.start()
        self.worker.update_progress.connect(self.run_cctv_update_progress)
        self.worker.update_lane_progress.connect(self.update_seg_lane)

    def estimate_save_click(self):
        self.worker.pause_flag = True
        self.worker.update_lane_progress.connect(self.update_seg_lane)
        self.worker.stop_flag = True

    def update_seg_lane(self, img_dict):
        final_sam_img = img_dict['final_sam_img']
        final_sam_img_pixmap = self.cvt_num_rgb_img_to_qtpixmap(final_sam_img)
        final_lane_img = img_dict['final_lane_img']
        final_lane_img_pixel = self.cvt_num_rgb_img_to_qtpixmap(final_lane_img)
        self.segany_label.setPixmap(final_sam_img_pixmap)
        self.lane_label.setPixmap(final_lane_img_pixel)

        save_result_pth = 'results/lane/'
        cv2.imwrite(save_result_pth + img_dict['vid_name'] + '_lane.jpg', img_dict['final_sam_img'])
        eval_log = (
            "Number of detected lanes: \t %d\n" % img_dict['no_detected_lanes'] +
            "Number of True Positive Lane:\t %d\n" % img_dict['final_tp'] +
            "Number of False Positive Lane:\t %d\n" % img_dict['final_fp'] +
            "Number of False Negative Lane:\t %d\n" % img_dict['final_fn'] +
            ("Precision - Recall - F1: \t {P}% - {R}% - {F1}%\n").format(P=img_dict['final_P']*100, R=img_dict['final_R']*100, F1=round(img_dict['final_f1']*100, 2))
            )
        eval_file = open(save_result_pth + img_dict['vid_name'] + '_lane.txt', "w")  # append mode
        eval_file.write(eval_log)
        eval_file.close()

        self.console_log_QTB.setText(eval_log)

    def run_cctv_update_progress(self, img_dict):
        frame_img = img_dict['frame']
        frame_img_pixmap = self.cvt_num_rgb_img_to_qtpixmap(frame_img)
        bg_img = img_dict['background_colored']
        bg_img_pixel = self.cvt_num_rgb_img_to_qtpixmap(bg_img)

        grid_ap_img = img_dict['grid_ap_show']
        grid_ap_img_pixel = self.cvt_num_rgb_img_to_qtpixmap(grid_ap_img)
        grid_mask_img = img_dict['grid_mask']
        grid_mask_img_pixel = self.cvt_num_rgb_img_to_qtpixmap(grid_mask_img)

        self.frame_label.setPixmap(frame_img_pixmap)
        self.bg_label.setPixmap(bg_img_pixel)
        self.grid_label.setPixmap(grid_ap_img_pixel)
        self.lane_roi_label.setPixmap(grid_mask_img_pixel)
        self.console_log_QTB.setText(img_dict['frame_log'])

    def run_all_dataset(self):
        self.run_all_flag = True
        self.worker = WorkerThread(self.cur_vid_pth, run_all_flag=self.run_all_flag)
        self.worker.start()
        self.worker.update_progress.connect(self.run_cctv_update_progress)
        self.worker.update_lane_progress.connect(self.update_seg_lane)


def main_demo():
    app = QApplication([])
    window = MyDemo()
    window.show()
    app.exec()


if __name__ == '__main__':
    main_demo()
