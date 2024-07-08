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

import cv2
import numpy as np
import random
import math
from scipy import stats


class Lane_Divider_Module():
	def __init__(self) -> None:
		print('Run lane divider')


	def find_dist_to_line(self, point, line):
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


	def display_vanishing_point(self, frame, vanishing_point):
		cv2.circle(frame, vanishing_point, 3, (0, 255, 0), 3, cv2.LINE_AA)
		return frame


	def get_inliers_w_vanishing_point(self, vp, lines, threshold_dist2vp=10):
		inliers = []

		for line in lines:
			dist = self.find_dist_to_line(vp, line)
			if (dist < threshold_dist2vp):
				inliers.append(line)

		return inliers


	def get_inliers_w_vanishing_point_catersian(self, vp, lines, threshold_dist2vp=10):
		inliers = []

		for line in lines:
			dist = self.find_dist_to_line_catersian(vp, line)
			if (dist < threshold_dist2vp):
				inliers.append(line)

		return inliers


	def find_intersection_point(self, line1, line2):
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


	def RANSAC(self, lines, ransac_iterations, ransac_threshold, ransac_ratio, ap_top):
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
			intersection_point = self.find_intersection_point(line1, line2)

			if intersection_point is not None:
				if (intersection_point[1] <= ap_top):
					# count the number of inliers num
					inlier_count = 0
					# inliers are lines whose distance to the point is less than ransac_threshold
					for line in lines:
						# find the distance from the line to the point
						dist = self.find_dist_to_line(intersection_point, line)
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


	def find_intersection_point_catersian(self, line1, line2):
		line11 = line1[:2]
		line22 = line2[:2]
		xdiff = (line11[0][0] - line11[1][0], line22[0][0] - line22[1][0])
		ydiff = (line11[0][1] - line11[1][1], line22[0][1] - line22[1][1])

		def det(a, b):
			return a[0] * b[1] - a[1] * b[0]

		div = det(xdiff, ydiff)
		if div == 0:
			# raise Exception('lines do not intersect')
			return None

		d = (det(*line11), det(*line22))
		x = det(d, xdiff) / div
		y = det(d, ydiff) / div
		return [x, y]


	def find_dist_to_line_catersian(self, point, line):
		x_diff = line[1][0] - line[0][0]
		y_diff = line[1][1] - line[0][1]
		x2y1 = line[1][0] * line[0][1]
		x1y2 = line[0][0] * line[1][1]
		b = np.sqrt(x_diff * x_diff + y_diff * y_diff)
		a = np.abs(y_diff * point[0] - x_diff * point[1] + x2y1 - x1y2)
		return a / b


	def line_equation_from_2_points(self, Q, P):
		a = Q[1] - P[1]
		b = P[0] - Q[0]
		c = a * (P[0]) + b * (P[1])

		return a, b, c


	def extend_lines(self, lines, max_x=1280, max_y=720):
		extended_lines = []
		for line in lines:
			a, b, c = self.line_equation_from_2_points(line[0], line[1])
			top_x = 0
			bot_x = max_x - 1
			if a != 0:
				top_x = c / a
				bot_x = (c - b * (max_y - 1)) / a
			# if 0 <= top_x < max_x and 0 <= bot_x < max_x:
			extended_lines.append([[top_x, 0], [bot_x, max_y - 1], line[0], line[1]])
		return extended_lines


	def RANSAC_catersian_form(self, lines, ransac_iterations, ransac_threshold, ransac_ratio, ap_top, extend_flag=False):
		inlier_count_ratio = 0.
		vanishing_point = (0, 0)
		if extend_flag:
			lines = self.extend_lines(lines)

		# perform RANSAC iterations for each set of lines
		for iteration in range(ransac_iterations):
			# randomly sample 2 lines
			n = 2
			selected_lines = random.sample(lines, n)
			line1 = selected_lines[0]
			line2 = selected_lines[1]
			# print(line1, line2)
			intersection_point = self.find_intersection_point_catersian(line1, line2)

			if intersection_point is not None:
				if (intersection_point[1] <= ap_top):
					# count the number of inliers num
					inlier_count = 0
					# inliers are lines whose distance to the point is less than ransac_threshold
					for line in lines:
						# find the distance from the line to the point
						dist = self.find_dist_to_line_catersian(intersection_point, line)
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
		return vanishing_point, lines

	def RANSAC_catersian_form_best(self, lines, ransac_iterations, ransac_threshold, ransac_ratio, ap_top, extend_flag=False):
		if extend_flag:
			lines = self.extend_lines(lines)
		best_ratio = 0
		best_vp = (0, 0)
		no_lines = len(lines)
		for idx_1 in range(0, no_lines):
			for idx_2 in range(0, no_lines):
				if idx_1 != idx_2:
					line1 = lines[idx_1]
					line2 = lines[idx_2]
					intersection_point = self.find_intersection_point_catersian(line1, line2)

					if intersection_point is not None:
						if (intersection_point[1] <= ap_top):
							# count the number of inliers num
							inlier_count = 0
							# inliers are lines whose distance to the point is less than ransac_threshold
							for line in lines:
								# find the distance from the line to the point
								dist = self.find_dist_to_line_catersian(intersection_point, line)
								# check whether it's an inlier or not
								if dist < ransac_threshold:
									inlier_count += 1

							# If the value of inlier_count is higher than previously saved value, save it, and save the current point
							inlier_ratio = inlier_count / no_lines

							if inlier_ratio > best_ratio:
								best_ratio = inlier_ratio
								best_vp = intersection_point

							# We are done in case we have enough inliers
							if inlier_ratio > ransac_ratio:
								return best_vp, lines

		return best_vp, lines

	def get_top_ap(self, grid_ap):
		top_x = 2480
		for ap in grid_ap:
			if (ap[1] < top_x):
				top_x = ap[1]

		return top_x


	def get_valid_lines(self, lines):
		valid_lines = []
		# Remove horizontal and vertical lines as they would not converge to vanishing point
		for line in lines:
			rho, theta = line[0]
			valid_lines.append(line)

		return valid_lines


	def display_lines_Rho_Theta(self, frame, lines):
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


	def generate_sam_mask(self, sam_masks, keep_mask_list, grid_mask, bg_img):
		final_sam_mask = np.zeros((bg_img.shape[0], bg_img.shape[1]))

		for idx_m in keep_mask_list:
			m = sam_masks[idx_m]['segmentation']
			m_area = sam_masks[idx_m]['area']
			value_grid = np.sum(m * grid_mask)

			if (value_grid > 0 and m_area > 20):
				final_sam_mask[m] = 1

		final_sam_img = (final_sam_mask * 255).astype(np.uint8)
		return final_sam_img


	def upgrade_sam_mask(self, sam_masks, keep_mask_list, grid_mask, bg_img):

		sam_mask = self.generate_sam_mask(sam_masks, keep_mask_list, grid_mask, bg_img)

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


	def angle_between_three_points(self, p1, p2, p3):

		a = p2[0] - p1[0]
		b = p2[1] - p1[1]
		c = p2[0] - p3[0]
		d = p2[1] - p3[1]

		aTanA = np.arctan2(a, b)
		aTanB = np.arctan2(c, d)

		return aTanB - aTanA


	def distance_between_2_points(self, p1, p2):
		return np.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))


	def DBSCAN_test(self, line_dists):
		# from sklearn.cluster import KMeans
		from sklearn.cluster import DBSCAN
		from sklearn.preprocessing import MinMaxScaler
		from collections import defaultdict

		extracted_line_infor = np.zeros((len(line_dists), 1))
		for idx in range(0, len(line_dists)):
			extracted_line_infor[idx][0] = line_dists[idx]

		scaler = MinMaxScaler()
		scaler.fit(extracted_line_infor)
		extracted_line_infor = scaler.fit_transform(extracted_line_infor)

		db = DBSCAN(eps=0.05, min_samples=1).fit(
			extracted_line_infor)  # applying DBSCAN Algorithm on our normalized lines
		labels = db.labels_

		extracted_line_infor = scaler.inverse_transform(extracted_line_infor)  # getting back our original values

		num_clusters = np.max(labels) + 1
		print(num_clusters, "clusters detected")

		grouped = defaultdict(list)
		# grouping lines by clusters
		for i, label in enumerate(labels):
			grouped[label].append(line_dists[i])

		# print('DB groups: ')
		# print(grouped[0])

		means = []
		# getting mean values by cluster
		for i in range(num_clusters):
			mean = np.mean(np.array(grouped[i]), axis=0)
			means.append(mean)
		means = np.array(means)

		return grouped, means


	def join_lane_lines(self, vp, lines, bg_img):

		h, w, c = bg_img.shape
		cx = w / 2
		cy = h / 2
		# cp = (cx, cy)
		cp = (0, cy)

		line_c = [[cy, 1.5708]]  # Horizontally Middle Line

		dist_list = np.zeros(len(lines))

		for idx_l in range(0, len(lines)):
			intersect_p = self.find_intersection_point(lines[idx_l], line_c)
			dist_list[idx_l] = intersect_p[0]

		grouped, means = self.DBSCAN_test(dist_list)
		# print(grouped)
		# print(means)

		bottom_means = np.zeros(len(means))

		for idx_m in range(0, len(means)):
			mean = means[idx_m]
			# print('mean: ', mean)
			p_mean = (mean, cy)
			p_2 = (vp[0], h - 1)

			p_vpm = (vp[0], cy)

			d2 = np.abs(cy - vp[1])
			d3 = self.distance_between_2_points(vp, p_mean)
			# print('d2, d3: ', d2, d3)
			angle_alpla = np.arccos(d2 / d3)

			d2x = np.tan(angle_alpla) * np.abs((p_2[1] - vp[1]))
			sign_x = (p_mean[0] - p_vpm[0]) / np.abs(p_mean[0] - p_vpm[0])

			bottom_mean = p_2[0] + d2x * sign_x

			bottom_means[idx_m] = bottom_mean

		return grouped, means, bottom_means


	def join_lane_lines_catersian(self, vp, lines, bg_img):
		h, w, c = bg_img.shape
		cx = w / 2
		cy = h / 2
		# cp = (cx, cy)
		cp = (0, cy)

		line_c = [[0, cy], [w - 1, cy]]  # Horizontally Middle Line

		dist_list = np.zeros(len(lines))

		for idx_l in range(0, len(lines)):
			intersect_p = self.find_intersection_point_catersian(lines[idx_l], line_c)
			dist_list[idx_l] = intersect_p[0]

		grouped, means = self.DBSCAN_test(dist_list)
		# print(grouped)
		# print(means)

		bottom_means = np.zeros(len(means))

		for idx_m in range(0, len(means)):
			mean = means[idx_m]
			# print('mean: ', mean)
			p_mean = (mean, cy)
			p_2 = (vp[0], h - 1)

			p_vpm = (vp[0], cy)

			d2 = np.abs(cy - vp[1])
			d3 = self.distance_between_2_points(vp, p_mean)
			# print('d2, d3: ', d2, d3)
			angle_alpla = np.arccos(d2 / d3)

			d2x = np.tan(angle_alpla) * np.abs((p_2[1] - vp[1]))
			sign_x = (p_mean[0] - p_vpm[0]) / np.abs(p_mean[0] - p_vpm[0])

			bottom_mean = p_2[0] + d2x * sign_x

			bottom_means[idx_m] = bottom_mean

		return grouped, means, bottom_means


	def join_lane_lines_v2(self, vp, lines, bg_img):

		h, w, c = bg_img.shape
		cx = w / 2
		cy = h / 2
		# cp = (cx, cy)
		cp = (0, cy)

		line_c = [[cy, 1.5708]]  # Horizontally Middle Line

		dist_list = np.zeros(len(lines))

		for idx_l in range(0, len(lines)):
			intersect_p = self.find_intersection_point(lines[idx_l], line_c)
			dist_list[idx_l] = intersect_p[0]

		no_max_lane = 12
		min_x = np.min(dist_list)
		max_x = np.max(dist_list)
		range_x = (max_x - min_x) / (no_max_lane - 1)

		draw_img = bg_img.copy()
		pb_lanes = np.zeros((no_max_lane, 3), dtype=np.float32)
		for lane_x in range(0, no_max_lane):
			pb_lanes[lane_x][0] = min_x + (lane_x * range_x)
			cv2.circle(draw_img, (int(min_x + (lane_x * range_x)), int(cy)), 10, (255, 255, 0), -1)

		for dist_x in dist_list:
			pb_lane_idx = int((dist_x - min_x) / range_x)
			pb_lanes[pb_lane_idx][1] += dist_x
			pb_lanes[pb_lane_idx][2] += 1

		print(pb_lanes)

		for lane_x in range(0, no_max_lane):
			if pb_lanes[lane_x][2] > 0:
				pb_lane_x = pb_lanes[lane_x][1] / pb_lanes[lane_x][2]
				cv2.circle(draw_img, (int(pb_lane_x), int(cy)), 5, (0, 255, 0), -1)

		# cv2.imshow('testing new', draw_img)

	def display_lane(self, vp, p_lines, pb_lines, img):
		h, w, c = img.shape
		for pline, pb_line in zip(p_lines, pb_lines):
			# cv2.line(img, vp, (int(pline), int(h / 2)), (255, 255, 0), 3, cv2.LINE_AA)
			cv2.line(img, vp, (int(pb_line), h - 1), (0, 255, 255), 3, cv2.LINE_AA)

		return img


	def find_center_of_shortest_edges(self, box):
		min_dist = 1000
		min_cx = -1
		min_cy = -1
		for idx in range(-1, 3):
			dist_idx = self.distance_between_2_points(box[idx], box[idx + 1])
			if dist_idx < min_dist:
				min_dist = dist_idx
				min_cx = int( (box[idx][0] + box[idx + 1][0]) / 2)
				min_cy = int( (box[idx][1] + box[idx + 1][1]) / 2)

		return [min_cx, min_cy]


	def run(self, sam_masks, keep_mask_list, grid_ap, grid_mask, bg_img):
		final_sam_img = self.upgrade_sam_mask(sam_masks, keep_mask_list, grid_mask, bg_img)
		# cv2.imshow('final_sam_img', final_sam_img)

		final_sam_img_gray = cv2.cvtColor(final_sam_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)

		_, final_sam_img_mask = cv2.threshold(final_sam_img_gray, 150, 255, cv2.THRESH_BINARY)
		mask_white_mask_canny = cv2.Canny(final_sam_img_mask, 1, 10, None, 3)
		# cv2.imshow('Canny', mask_white_mask_canny)
		kernel = np.ones((1, 1), np.uint8)
		mask_white_mask_canny = cv2.dilate(mask_white_mask_canny, kernel, iterations=2)
		mask_white_mask_canny = cv2.erode(mask_white_mask_canny, kernel, iterations=1)

		# lines_RT = cv2.HoughLines(mask_white_mask_canny, 1, np.pi / 360, 50, None, 0, 0)
		lines_RT = cv2.HoughLines(mask_white_mask_canny, 1, np.pi / 360, 30, None, 0, 0)
		if (lines_RT is not None):
			# grouped_lines, means_lines = group_lines_by_rho(lines_RT)
			# show_lines_by_group(grouped_lines, bg_img.copy())
			valid_lines = self.get_valid_lines(lines_RT)

			# print('Number of lines: ', len(valid_lines))
			hough_RT_visual = self.display_lines_Rho_Theta(bg_img.copy(), valid_lines)
			# cv2.imshow('hough_RT_visual', hough_RT_visual)

			# RANSAC parameters:
			# ransac_iterations,ransac_threshold,ransac_ratio = 350, 10, 0.90
			ransac_iterations, ransac_threshold, ransac_ratio = 350, 15, 0.70
			ap_top = self.get_top_ap(grid_ap)
			# vanishing_point = RANSAC_2(valid_lines, ransac_iterations, ransac_threshold, ransac_ratio, ap_top)
			vanishing_point = self.RANSAC(valid_lines, ransac_iterations, ransac_threshold, ransac_ratio, ap_top)
			confirmed_lines = self.get_inliers_w_vanishing_point(vanishing_point, lines_RT)
			confirmed_lines_visual = self.display_lines_Rho_Theta(bg_img.copy(), confirmed_lines)
			confirmed_lines_visual = self.display_vanishing_point(confirmed_lines_visual, vanishing_point)


			if len(confirmed_lines) == 0:
				print('Hough can not find any lines')
				return final_sam_img, final_sam_img, (0, 0), [-1, -1]

			line_g, line_m, line_m_b = self.join_lane_lines(vanishing_point, confirmed_lines, bg_img)
			# self.join_lane_lines_v2(vanishing_point, confirmed_lines, bg_img)
			confirmed_lines_visual = self.display_lane(vanishing_point, line_m, line_m_b, confirmed_lines_visual)
			# print(line_g, line_m)

			# cv2.imshow("Final lane", confirmed_lines_visual)
			return final_sam_img, confirmed_lines_visual, vanishing_point, line_m_b
		else:
			print('Hough can not find any lines')
			return final_sam_img, final_sam_img, (0, 0), [-1, -1]

	def is_subset(self, a, b):
		b = np.unique(b)
		c = np.intersect1d(a, b)
		return c.size == b.size

	def merge_lines(self, line_list, contour_img):
		extend_line_list = self.extend_lines(line_list)
		matching_mat = np.zeros((len(extend_line_list), len(extend_line_list)))
		for extend_idx, extend_line in enumerate(extend_line_list):
			# contour_img = cv2.line(contour_img, (int(extend_line[0][0]), int(extend_line[0][1])),
			# 					   (int(extend_line[1][0]), int(extend_line[1][1])), (255, 255, 0), 2)
			extend_cX_line = (extend_line[0][0] + extend_line[1][0]) / 2
			extend_cY_line = (extend_line[0][1] + extend_line[1][1]) / 2
			extend_cp_line = [extend_cX_line, extend_cY_line]
			dist_to_line = []

			for idx_normal, line in enumerate(line_list):
				center_p = line[0]
				a_dist = self.find_dist_to_line_catersian(center_p, extend_line)
				if a_dist < 5:
					matching_mat[extend_idx][idx_normal] = 1
					contour_img = cv2.circle(contour_img, (int(center_p[0]), int(center_p[1])),
												  10, (255, 0, 255), 1)
				dist_to_line.append(a_dist)


		no_lines = len(line_list)
		matching_mat_group = []


		for idx in range(0, no_lines):
			match_list = np.where(matching_mat[idx] == 1)
			matching_mat_group.append(match_list)

		# print(matching_mat_group)

		merging_stt = np.full(no_lines, 1)
		for idx_a in range(0, no_lines):
			if merging_stt[idx_a] == 1:
				list_a = matching_mat_group[idx_a]
				# print('List a: ', list_a)
				for idx_b in range(0, no_lines):
					if idx_a != idx_b:
						list_b = matching_mat_group[idx_b]
						if self.is_subset(list_a, list_b):
							merging_stt[idx_b] = 0
		remain_idx = np.where(merging_stt==1)[0]
		remain_matching_group = [matching_mat_group[i] for i in remain_idx]
		return remain_matching_group, contour_img

	def run_develop(self, sam_masks, keep_mask_list, grid_ap, grid_mask, bg_img):
		final_sam_img = self.upgrade_sam_mask(sam_masks, keep_mask_list, grid_mask, bg_img)
		# cv2.imshow('final_sam_img', final_sam_img)

		final_sam_img_gray = cv2.cvtColor(final_sam_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)

		_, final_sam_img_mask = cv2.threshold(final_sam_img_gray, 150, 255, cv2.THRESH_BINARY)
		mask_white_mask_canny = cv2.Canny(final_sam_img_mask, 1, 10, None, 3)
		cv2.imshow('Canny', mask_white_mask_canny)
		# mask_white_mask_canny_copy = mask_white_mask_canny.copy()

		kernel = np.ones((1, 1), np.uint8)
		mask_white_mask_canny = cv2.dilate(mask_white_mask_canny, kernel, iterations=2)
		mask_white_mask_canny = cv2.erode(mask_white_mask_canny, kernel, iterations=1)

		contours, hierarchy = cv2.findContours(mask_white_mask_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contour_img = cv2.drawContours(bg_img.copy(), contours, -1, (0, 255, 0), 3)

		lines_catersian_2_points = []
		for c in contours:
			M = cv2.moments(c)
			if M["m00"] != 0:
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])
				contour_img = cv2.circle(contour_img, (cX, cY), 3, (0, 0, 255), -1)
				rect = cv2.minAreaRect(c)
				box = cv2.boxPoints(rect)
				box = np.int0(box)
				se_center = self.find_center_of_shortest_edges(box)
				contour_img = cv2.drawContours(contour_img,[box], 0, (255,0,0), 2)
				contour_img = cv2.line(contour_img, (cX, cY), (se_center[0], se_center[1]), (255,255,0), 2)
				lines_catersian_2_points.append([[cX, cY], [se_center[0], se_center[1]]])

		group_lines, contour_img = self.merge_lines(lines_catersian_2_points, contour_img)
		joined_lines = []
		for group_line in group_lines:
			contour_img_check = contour_img.copy()
			# print(group_line)
			for line_idx in group_line[0]:
				center_p = lines_catersian_2_points[line_idx][0]
				contour_img_check = cv2.circle(contour_img_check, (int(center_p[0]), int(center_p[1])),
																20, (0, 0, 255), 1)

			cur_selected_lines = np.array([lines_catersian_2_points[i] for i in group_line[0]])
			# print(cur_selected_lines.shape)
			x_arr = cur_selected_lines[:,0,0]
			y_arr = cur_selected_lines[:,0,1]
			slope, intercept, r_value, p_value, std_err = stats.linregress(x_arr, y_arr)
			if math.isnan(slope) is False:
				# print(slope, intercept, r_value, p_value)
				# z = np.polyfit(x_arr, y_arr, 1)
				x_top = -intercept / slope
				x_bot = ((720 - 1) - intercept) / slope
				contour_img_check = cv2.line(contour_img_check, (int(x_top), 0), (int(x_bot), 720-1), (255,255,0), 2)
				joined_lines.append([[x_top, 0], [x_bot, 720-1]])

		ransac_iterations, ransac_threshold, ransac_ratio = 350, 10, 0.95
		ap_top = self.get_top_ap(grid_ap)
		vanishing_point, extended_lines = self.RANSAC_catersian_form_best(joined_lines, ransac_iterations, ransac_threshold, ransac_ratio, ap_top)
		confirmed_lines = self.get_inliers_w_vanishing_point_catersian(vanishing_point, extended_lines)
		for confirmed_line in confirmed_lines:
			# print(confirmed_line)
			contour_img = cv2.line(contour_img, (int(confirmed_line[0][0]), int(confirmed_line[0][1])),
									   (int(confirmed_line[1][0]), int(confirmed_line[1][1])), (255, 255, 0), 2)
		print('vp: ', vanishing_point, len(confirmed_lines))
		cv2.imshow('Contour', contour_img)

		if len(confirmed_lines) == 0:
			print('Hough can not find any lines')
			return contour_img, contour_img, (0, 0), None

		return contour_img, contour_img, vanishing_point, confirmed_lines

		# else:
		# 	print('Hough can not find any lines')
		# 	return final_sam_img, final_sam_img, (0, 0), [-1, -1]
