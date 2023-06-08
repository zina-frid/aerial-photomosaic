import cv2
import numpy as np
from exif import Image as eImage
from image_processing.get_image_data import get_coordinates
from math import sqrt
import pyproj


def calculate_distance_in_meters(lat1, lon1, lat2, lon2):
    # Create UTM projection
    utm = pyproj.Proj(proj='utm', zone='10N')  # Adjust the zone according to your location

    # Convert geographic coordinates to UTM coordinates
    x1, y1 = utm(lon1, lat1)
    x2, y2 = utm(lon2, lat2)

    # Calculate Euclidean distance in meters
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    return distance

def calculate_overlap_percentage_with_alignment(image1_path, image2_path, coordinates1, coordinates2, max_distance_threshold):
    # Load images
    image1 = cv2.imread(image1_path, 0)
    image2 = cv2.imread(image2_path, 0)
    resize = True

    if resize:
        percent = 20  # percent of original size
        dim = (int(image1.shape[1] * percent / 100), int(image1.shape[0] * percent / 100))
        image1 = cv2.resize(image1, dim, interpolation=cv2.INTER_AREA)
        dim = (int(image2.shape[1] * percent / 100), int(image2.shape[0] * percent / 100))
        image2 = cv2.resize(image2, dim, interpolation=cv2.INTER_AREA)

    # Create ORB detector
    detector = cv2.BRISK_create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = detector.detectAndCompute(image1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(image2, None)

    # Match keypoints
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)

    # Filter matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:50]  # Choose top 50 matches or adjust the number based on your requirement

    # Extract matched keypoints
    matched_keypoints1 = np.float32([keypoints1[match.queryIdx].pt for match in good_matches])
    matched_keypoints2 = np.float32([keypoints2[match.trainIdx].pt for match in good_matches])

    # Calculate homography matrix
    H, _ = cv2.findHomography(matched_keypoints2, matched_keypoints1, cv2.RANSAC, 5.0)

    # Calculate Euclidean distance between coordinates
    distance = sqrt((coordinates1[0] - coordinates2[0]) ** 2 + (coordinates1[1] - coordinates2[1]) ** 2)

    # Check if distance exceeds the threshold
    if distance > max_distance_threshold:
        return 0.0  # Return 0% overlap if distance is too large

    # Warp image2 to align with image1
    aligned_image2 = cv2.warpPerspective(image2, H, (image1.shape[1], image1.shape[0]))

    # Calculate overlap percentage
    overlap_pixels = cv2.bitwise_and(image1, aligned_image2)
    overlap_count = np.count_nonzero(overlap_pixels)
    total_pixels = image1.shape[0] * image1.shape[1]
    overlap_percentage = (overlap_count / total_pixels) * 100

    return overlap_percentage


image1_path = ""
image2_path = ""
image_read = open(image1_path, 'rb')
image_1_object = eImage(image_read)
lat1, lon1 = get_coordinates(image_1_object)
image_read = open(image2_path, 'rb')
image_2_object = eImage(image_read)
lat2, lon2 = get_coordinates(image_2_object)
max_distance_threshold = 0.001
overlap_percentage = calculate_overlap_percentage_with_alignment(image1_path, image2_path, (lat1, lon1), (lat2, lon2),max_distance_threshold )
print(f"Процент перекрытия с учетом смещения: {overlap_percentage}%")