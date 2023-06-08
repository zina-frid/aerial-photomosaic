from image_processing.stitch_images import stitch, stitch_sift
from datetime import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt

directory = ""
img_name_1 = ''
img_name_2 = ''


def calculate_rmse(original_image1, original_image2, stitched_image):
    
    img_original1 = cv2.imread(original_image1)
    img_original2 = cv2.imread(original_image2)
    img_stitched = cv2.imread(stitched_image)

    # Resize the stitched image to match the original images
    img_stitched_resized = cv2.resize(img_stitched, (img_original1.shape[1], img_original1.shape[0]))

    # Check for matching image sizes
    assert img_original1.shape == img_original2.shape == img_stitched_resized.shape, "Изображения имеют разные размеры."

    # Difference between original images and stitched image
    diff1 = cv2.absdiff(img_original1, img_stitched_resized)
    diff2 = cv2.absdiff(img_original2, img_stitched_resized)

    # Squared pixel difference
    squared_diff1 = np.square(diff1)
    squared_diff2 = np.square(diff2)

    # Sum of squared pixel difference
    sum_squared_diff1 = np.sum(squared_diff1)
    sum_squared_diff2 = np.sum(squared_diff2)

    # Average of the sum of squares of the pixel difference
    mean_squared_diff1 = sum_squared_diff1 / (img_stitched_resized.shape[0] * img_stitched_resized.shape[1] * img_stitched_resized.shape[2])
    mean_squared_diff2 = sum_squared_diff2 / (img_stitched_resized.shape[0] * img_stitched_resized.shape[1] * img_stitched_resized.shape[2])

    # Root of the mean of the sum of squares of the pixel difference
    rmse1 = np.sqrt(mean_squared_diff1)
    rmse2 = np.sqrt(mean_squared_diff2)

    return rmse1, rmse2


def calculate_ioc(original_image1, original_image2, stitched_image):
    
    image1 = cv2.imread(original_image1)
    image2 = cv2.imread(original_image2)
    stitched_image = cv2.imread(stitched_image)

    # Resize the stitched image to match the original images
    stitched_image = cv2.resize(stitched_image, (image1.shape[1], image1.shape[0]))

    # Check for matching image sizes
    assert image1.shape == image2.shape == stitched_image.shape, "Изображения имеют разные размеры."

    # Convert images to grayscale
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gray_stitched_image = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)

    # Calculate IoC for each pixel
    ioc_map = np.abs(gray_image1 - gray_stitched_image) + np.abs(gray_image2 - gray_stitched_image)
    ioc = np.mean(ioc_map)

    return ioc


def stitching_quality_assessment():
    original_image1 = directory + '\\' + img_name_1
    original_image2 = directory + '\\' + img_name_2
    stitched_image = directory + '\\' + 'stitched_brisk.png'

    # BRISK
    print("----------BRISK----------")
    brisk_result = stitch(directory, "brisk", resize_dataset, resize_result)
    brisk_result.save(directory + '\\' + "stitched_brisk" + ".png")
    brisk_rmse1, brisk_rmse2 = calculate_rmse(original_image1, original_image2, stitched_image)
    print("RMSE between original image 1 and stitched image:", brisk_rmse1)
    print("RMSE between original image 2 and stitched image:", brisk_rmse2)

    # Расчет IoC
    brisk_ioc = calculate_ioc(original_image1, original_image2, stitched_image)
    print("IoC value:", brisk_ioc)


    # SIFT
    print("----------SIFT----------")
    sift_result = stitch_sift(directory, resize_dataset, resize_result)
    sift_result.save(directory + '\\' + "stitched_sift" + ".png")
    stitched_image = directory + '\\' + 'stitched_sift.png'
    sift_rmse1, sift_rmse2 = calculate_rmse(original_image1, original_image2, stitched_image)
    print("RMSE between original image 1 and stitched image:", sift_rmse1)
    print("RMSE between original image 2 and stitched image:", sift_rmse2)

    # Расчет IoC
    sift_ioc = calculate_ioc(original_image1, original_image2, stitched_image)
    print("IoC value:", sift_ioc)


    return brisk_rmse1, brisk_rmse2, sift_rmse1, sift_rmse2, [brisk_ioc, sift_ioc]


resize_dataset = (False, 0)  # min 5
resize_result = (True, 20)
x_array = []
brisk_array, sift_array, orb_array, ioc_array = [], [], [], []
for i in range(10, 1, -1):
    print(f'{i * 10}%')
    resize_dataset = (True, i * 10)
    x_array.append(i * 10)
    brisk_rmse1, brisk_rmse2, sift_rmse1, sift_rmse2, ioc = stitching_quality_assessment()
    brisk_array.append([brisk_rmse1, brisk_rmse2])
    sift_array.append([sift_rmse1, sift_rmse2])
    ioc_array.append(ioc)

# brisk
fig, ax = plt.subplots()

brisk_first = [brisk[0] for brisk in brisk_array]
ax.plot(x_array, brisk_first, color='green', label='RMSE 1')
brisk_second = [brisk[1] for brisk in brisk_array]
ax.plot(x_array, brisk_second, linestyle='dashed', color='green',  label='RMSE 2')
ax.scatter(x_array, brisk_first, color='green')
ax.scatter(x_array, brisk_second, color='green')
ax.set_xlabel('Процент от исходного изображения')
ax.set_ylabel('rsme')
ax.set_title('График зависимости rsme для BRISK')
ax.legend()
plt.show()

# sift
fig_, ax = plt.subplots()

sift_first = [sift[0] for sift in sift_array]
ax.plot(x_array, sift_first, color='orange', label='RMSE 1')
sift_second = [sift[1] for sift in sift_array]
ax.plot(x_array, sift_second, linestyle='dashed', color='orange', label='RMSE 2')
ax.scatter(x_array, sift_first, color='orange')
ax.scatter(x_array, sift_second, color='orange')

ax.set_xlabel('Процент от исходного изображения')
ax.set_ylabel('rsme')
ax.set_title('График зависимости rsme для SIFT')
ax.legend()
plt.show()

# ioc
_, ax = plt.subplots()

first, second = [i for i in ioc_array]
ax.plot(x_array, first, color='green', label='BRISK IoC')
ax.plot(x_array, second, color='orange', label='SIFT IoC')
ax.scatter(x_array, first, color='green')
ax.scatter(x_array, second, color='orange')
ax.set_xlabel('Процент от исходного изображения')
ax.set_ylabel('IoC')
ax.set_title('График зависимости IoC')
ax.legend()
plt.show()