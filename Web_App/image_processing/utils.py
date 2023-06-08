import glob
import os
import shutil

import cv2
import numpy
from datetime import datetime
from image_processing.stitch_images import stitch
from image_processing.map_layer import generate_map, make_map
from image_processing.get_image_data import create_dataframe
from image_processing.rotation import rotate
from image_processing.generate_tiles import create_image_tiles

directory = './server/uploaded_images'
method = "brisk"


# Cleaning up the working directory
def clear_working_directory():
    file_list = glob.glob((directory + '\\' + "*"))
    # for f in file_list:
    #     os.remove(f)
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Ошибка при удалении {file_path}: {e}")


# Creating a transparent background
def transparent(result):
    img = numpy.array(result)
    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Creating a Threshold
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    # Color image channel separation
    b, g, r = cv2.split(img)
    rgba = [b, g, r, alpha]
    # Combining rgba into a multichannel image
    dst = cv2.merge(rgba, 4)
    cv2.imwrite(directory + '\\' + "result_mosaic" + ".png", dst)


# Removing uploaded images from working directory
def remove_images_except_res():
    file_list = glob.glob((directory + '\\' + "*"))
    file_list.remove(directory + '\\' + 'result_mosaic.png')
    file_list.remove(directory + '\\' + 'map.html')
    for f in file_list:
        os.remove(f)


# Starting staged processing
def processing():
    global directory, method
    resize_dataset = (True, 20)  # min 5
    resize_result = (False, 20)

    start = datetime.now()
    print(f"Time start: {start}")

    # read exif
    images_df, image_dict = create_dataframe(directory)
    print(f"image_df: {images_df}")
    print(f"image_dict: {image_dict}")

    # stitch images
    stitched_result = stitch(directory, method, resize_dataset, resize_result)
    stitched_result.save(directory + '\\' + "stitched" + ".png")
    print("stitch done!")

    # angle
    rotated_result, angle = rotate(stitched_result, images_df)
    rotated_result.save(directory + '\\' + "rotated" + ".png")
    print("rotate done!")

    # transparent
    transparent(rotated_result)
    print("transparent done!")

    # tiles
    create_image_tiles(directory)

    # display on map
    size = stitched_result.size
    size_r = rotated_result.size
    generate_map(images_df, directory, size_r)
    # make_map(images_df, angle, directory, size)
    print("map done!")


    end = datetime.now() - start
    print(f"Duration: {end}")
