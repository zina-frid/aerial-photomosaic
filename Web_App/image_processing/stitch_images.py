# -*- coding: utf-8 -*-
import cv2
import numpy as np
import glob
from PIL import Image


def warp_images(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    # coordinates of a reference image
    list_of_points_1 = np.float32([
        [0, 0],
        [0, rows1],
        [cols1, rows1],
        [cols1, 0]
    ]).reshape(-1, 1, 2)
    # coordinates of second image
    temp_points = np.float32([
        [0, 0],
        [0, rows2],
        [cols2, rows2],
        [cols2, 0]
    ]).reshape(-1, 1, 2)

    # When we have established a homography we need to warp perspective
    # Change field of view
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)  # calculate the transformation matrix

    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img1

    return output_img


def stitch(input_directory, method, resize_dataset, resize_result):
    img_list = read_images(input_directory, resize_dataset)
    # Use ORB or BRISK detector to extract keypoints
    if method == "orb":
        detector = cv2.ORB_create(nfeatures=2000)
    else:
        detector = cv2.BRISK_create()

    while True:
        img1 = img_list.pop(0)
        img2 = img_list.pop(0)

        # Find the key points and descriptors
        keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
        keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

        print(f"Image 1: keypoints = {len(keypoints1)}, descriptors = {len(descriptors1)}")
        print(f"Image 2: keypoints = {len(keypoints2)}, descriptors = {len(descriptors2)}")

        # Create a BFMatcher object to match descriptors
        # It will find all of the matching keypoints on two images
        # NORM_HAMMING specifies the distance as a measurement of similarity between two descriptors
        bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

        # Find matching points
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        print(f"Found matches = {len(matches)}")

        all_matches = []
        for m, n in matches:
            all_matches.append(m)
        # Finding the best matches
        good = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:  # Threshold
                good.append(m)

        # Set minimum match condition
        min_match_count = 5

        print(f"Good matches = {len(good)}")
        if len(good) > min_match_count:

            # Convert keypoints to an argument for findHomography
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # Establish a homography
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            result = warp_images(img2, img1, M)
            img_list.insert(0, result)

            if len(img_list) == 1:
                break

    result = Image.fromarray(np.uint8(result)).convert('RGB')
    if resize_result[0]:
        percent = resize_result[1]
        width, height = result.size
        width = int(width * percent / 100)
        height = int(height * percent / 100)
        result = result.resize((width, height))
    return result


def read_images(input_directory, resize_dataset):
    # folder containing images from drones, sorted by name
    path = sorted(glob.glob((input_directory + '\\' + "*.jpg")))

    img_list = []
    for img in path:
        n = cv2.imread(img)
        if resize_dataset[0]:
            percent = resize_dataset[1]  # percent of original size
            width = int(n.shape[1] * percent / 100)
            height = int(n.shape[0] * percent / 100)
            dim = (width, height)
            n = cv2.resize(n, dim, interpolation=cv2.INTER_AREA)
        img_list.append(n)

    return img_list


# version with SIFT + FLANN
def stitch_sift(input_directory, resize_dataset, resize_result, min_match_count=5):
    img_list = read_images(input_directory, resize_dataset)

    sift = cv2.SIFT_create()
    while True:
        img1 = img_list.pop(0)
        img2 = img_list.pop(0)
        # Find the key points and descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

        print(f"Image 1: keypoints = {len(keypoints1)}, descriptors = {len(descriptors1)}")
        print(f"Image 2: keypoints = {len(keypoints2)}, descriptors = {len(descriptors2)}")

        # Initialize parameters for matching based on FLANN
        # (Fast Library for Approximate Nearest Neighbors)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        # Initialize FLANN
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Calculate matches
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        print(f"Found matches = {len(matches)}")

        # Store all the good matches as per Lowe's ratio test
        good_matches = []
        for m1, m2 in matches:
            if m1.distance < 0.6 * m2.distance:
                good_matches.append(m1)

        print(f"Good matches = {len(good_matches)}")

        if len(good_matches) > min_match_count:
            src_pts = np.float32([keypoints1[good_match.queryIdx].pt
                                  for good_match in good_matches]).reshape(-1, 1, 2)

            dst_pts = np.float32([keypoints2[good_match.trainIdx].pt
                                  for good_match in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            result = warp_images(img2, img1, M)
            img_list.insert(0, result)

            if len(img_list) == 1:
                break

    result = Image.fromarray(np.uint8(result)).convert('RGB')
    if resize_result[0]:
        percent = resize_result[1]
        width, height = result.size
        width = int(width * percent / 100)
        height = int(height * percent / 100)
        result = result.resize((width, height))
    return result
