from image_processing.stitch_images import warp_images
from datetime import datetime
import cv2
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt


def stitch(input_directory, method, resize_dataset, resize_result):
    img_list = read_images(input_directory, resize_dataset)
    # Use ORB or BRISK detector to extract keypoints
    if method == "orb":
        detector = cv2.ORB_create(nfeatures=2000)
    else:
        detector = cv2.BRISK_create()

    keypoints_num, matches_num, good_matches_num = (0, 0), 0, 0
    try:
        while True:
            img1 = img_list.pop(0)
            img2 = img_list.pop(0)
            # Find the key points and descriptors
            keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
            keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

            print(f"Image 1: keypoints = {len(keypoints1)}, descriptors = {len(descriptors1)}")
            print(f"Image 2: keypoints = {len(keypoints2)}, descriptors = {len(descriptors2)}")
            keypoints_num = (len(keypoints1), len(keypoints2))

            # Create a BFMatcher object to match descriptors
            # It will find all of the matching keypoints on two images
            # NORM_HAMMING specifies the distance as a measurement of similarity between two descriptors
            bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

            # Find matching points
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)

            print(f"Found matches = {len(matches)}")
            matches_num = len(matches)

            all_matches = []
            for m, n in matches:
                all_matches.append(m)
            # Finding the best matches
            good = []
            for m, n in matches:
                if m.distance < 0.6 * n.distance:  # Threshold
                    good.append(m)

            good_matches_num = len(good)

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
        return result, keypoints_num, matches_num, good_matches_num
    except Exception:
        return [], keypoints_num, matches_num, good_matches_num


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
    keypoints_num, matches_num, good_matches_num = (0, 0), 0, 0
    try:
        while True:
            img1 = img_list.pop(0)
            img2 = img_list.pop(0)
            # Find the key points and descriptors
            keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
            keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

            print(f"Image 1: keypoints = {len(keypoints1)}, descriptors = {len(descriptors1)}")
            print(f"Image 2: keypoints = {len(keypoints2)}, descriptors = {len(descriptors2)}")
            keypoints_num = (len(keypoints1), len(keypoints2))

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
            matches_num = len(matches)

            # Store all the good matches as per Lowe's ratio test
            good_matches = []
            for m1, m2 in matches:
                if m1.distance < 0.6 * m2.distance:
                    good_matches.append(m1)

            print(f"Good matches = {len(good_matches)}")
            good_matches_num = len(good_matches)

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
        return result, keypoints_num, matches_num, good_matches_num
    except Exception:
        return [], keypoints_num, matches_num, good_matches_num


def stitching_resize():
    # BRISK
    print("----------BRISK----------")
    start = datetime.now()
    print(f"Time start: {start}")
    brisk_result, br_kp, br_mat, br_good = stitch(directory, "brisk", resize_dataset, resize_result)
    print(brisk_result, br_kp, br_mat, br_good)
    brisk_end = datetime.now() - start
    print(f"Duration: {brisk_end}")

    # SIFT
    print("----------SIFT----------")
    start = datetime.now()
    print(f"Time start: {start}")
    sift_result, si_kp, si_mat, si_good = stitch_sift(directory, resize_dataset, resize_result)
    sift_end = datetime.now() - start
    print(f"Duration: {sift_end}")

    return br_kp, br_mat, br_good, si_kp, si_mat, si_good, [brisk_end.total_seconds(), sift_end.total_seconds()]


def make_graphics():
    # График КТ для 1го
    fig, ax = plt.subplots()

    brisk_k1, brisk_k2 = [brisk[0] for brisk in brisk_array]
    ax.plot(x_array, brisk_k1, color='green', label='brisk - снимок 1')
    ax.plot(x_array, brisk_k2, color='green', linestyle='dashed', label='brisk - снимок 2')
    sift_k1, sift_k2 = [sift[0] for sift in sift_array]
    ax.plot(x_array, sift_k1, color='orange', label='sift - снимок 1')
    ax.plot(x_array, sift_k2, color='orange', linestyle='dashed', label='sift - снимок 2')

    ax.scatter(x_array, brisk_k1, color='green')
    ax.scatter(x_array, brisk_k2, color='green')
    ax.scatter(x_array, sift_k1, color='orange')
    ax.scatter(x_array, sift_k2, color='orange')

    ax.set_xlabel('Процент от исходного изображения')
    ax.set_ylabel('Количество особых точек')
    ax.legend()
    plt.show()

    # График matches
    fi_, ax = plt.subplots()

    brisk_mat = [brisk[1] for brisk in brisk_array]
    ax.plot(x_array, brisk_mat, color='green', label='brisk')
    sift_mat = [sift[1] for sift in sift_array]
    ax.plot(x_array, sift_mat, color='orange', label='sift')

    ax.scatter(x_array, brisk_mat, color='green')
    ax.scatter(x_array, sift_mat, color='orange')

    ax.set_xlabel('Процент от исходного изображения')
    ax.set_ylabel('Количество совпадений')
    ax.legend()
    plt.show()

    # График good matches
    fi, ax = plt.subplots()
    brisk_good = [brisk[2] for brisk in brisk_array]
    ax.plot(x_array, brisk_good, color='green', label='brisk')
    sift_good = [sift[2] for sift in sift_array]
    ax.plot(x_array, sift_good, color='orange', label='sift')

    ax.scatter(x_array, brisk_good, color='green')
    ax.scatter(x_array, sift_good, color='orange')

    ax.set_xlabel('Процент от исходного изображения')
    ax.set_ylabel('Количество хороших совпадений')
    ax.legend()
    plt.show()

    # График time
    fi, ax = plt.subplots()
    print(end_time)
    br_end = [tm[0] for tm in end_time]
    si_end = [tm[1] for tm in end_time]
    ax.plot(x_array, br_end, color='green', label='brisk')
    ax.plot(x_array, si_end, color='orange', label='sift')
    ax.scatter(x_array, br_end, color='green')
    ax.scatter(x_array, si_end, color='orange')
    ax.set_xlabel('Процент от исходного изображения')
    ax.set_ylabel('Время сшивания, с')
    ax.legend()
    plt.show()


directory = ""
resize_dataset = (False, 0)  # min 5
resize_result = (True, 20)
x_array = []
brisk_array, sift_array, end_time = [], [], []
for i in range(10, 1, -1):
    print(f'{i * 10}%')
    resize_dataset = (True, i * 10)
    x_array.append(i * 10)
    br_kp, br_mat, br_good, si_kp, si_mat, si_good, end = stitching_resize()
    brisk_array.append([br_kp, br_mat, br_good])
    sift_array.append([si_kp, si_mat, si_good])
    end_time.append(end)
    print(end)

make_graphics()
