import os
from exif import Image
from image_processing.get_image_data import get_coordinates
import requests


def get_geodetic_height(latitude, longitude):
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={latitude},{longitude}"
    response = requests.get(url)
    data = response.json()
    elevation = data["results"][0]["elevation"]
    return elevation


def calculate_flight_height(latitude, longitude, gps_altitude):
    geodetic_height = get_geodetic_height(latitude, longitude)

    # Вычисляем высоту полета над земной поверхностью
    flight_height = gps_altitude - geodetic_height
    print(f'gps_altitude {gps_altitude}')
    print(f'geodetic_height {geodetic_height}')
    print(f'height {flight_height}')
    return flight_height


def calc_ratio(image_width, image_height, lat, lon, alt, focal_length):

    # flight_height = calculate_flight_height(lat, lon, alt)
    flight_height = 20
    sensor_width = 17.3
    sensor_height = 13.0
    gsd_w = (flight_height * sensor_width) / (focal_length * image_width)
    gsd_h = (flight_height * sensor_height) / (focal_length * image_height)
    return gsd_w, gsd_h


# Processing all images
# use a pandas dataframe for storing lats and longs
def start_calc():
    # Get list of files in directory
    images_dir = ''
    images_list = os.listdir(images_dir)
    # Filter list to get image files
    images_list = [i for i in images_list if i[-4:] == '.JPG']

    for image in images_list:
        image_path = os.path.join(images_dir, image)
        image_read = open(image_path, 'rb')
        image_object = Image(image_read)
        lat, lon = get_coordinates(image_object)
        alt = image_object.gps_altitude

        focal_length = image_object.focal_length
        pixel_x_dimension = image_object.pixel_x_dimension
        pixel_y_dimension = image_object.pixel_y_dimension
        gsd_w, gsd_h = calc_ratio(pixel_x_dimension, pixel_y_dimension, lat, lon, alt, focal_length)
        print(f'ширина: {pixel_x_dimension} пикселей, высота: {pixel_y_dimension} пикселей')
        print(f'ширина: {gsd_w} метров/пиксель, высота: {gsd_h} метров/пиксель')
