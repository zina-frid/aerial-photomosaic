import os
import exif
from image_processing.pixel_meter import calc_ratio
from geopy.distance import distance
from math import sin, radians
from PIL import Image
from image_processing.get_image_data import get_coordinates
from math import atan2, degrees


# Function to rotate image
def rotate_image(image, angle):
    # Rotate the image by the given angle
    rotated_image = image.rotate(angle, expand=True)
    return rotated_image


def calculate_slope_angle(lat1, lon1, lat2, lon2):
    # Calculate differences in coordinates
    lat_diff = abs(lat2 - lat1)
    lon_diff = abs(lon2 - lon1)

    # Calculate the angle between the straight line and the north-south direction
    angle = atan2(lon_diff, lat_diff)

    # Convert radians to degrees
    angle_degrees = degrees(angle)
    print(f'angle:{angle_degrees}')
    return angle_degrees


def calc_delta(directory, angle, size):
    image_path = os.path.join(directory, "0000.jpg")
    image_read = open(image_path, 'rb')
    image_object = exif.Image(image_read)
    lat, lon = get_coordinates(image_object)
    alt = image_object.gps_altitude
    alt_ref = image_object.gps_altitude_ref
    print(f'alt_ref: {alt_ref}')

    if alt_ref == 0 or str(alt_ref).lower() == 'above sea level':
        alt = alt
    elif alt_ref == 1 or str(alt_ref).lower() == 'below sea level':
        alt = abs(alt)
    else:
        alt = 0

    focal_length = image_object.focal_length
    pixel_x_dimension = image_object.pixel_x_dimension
    pixel_y_dimension = image_object.pixel_y_dimension

    gsd_w, gsd_h = calc_ratio(pixel_x_dimension, pixel_y_dimension, lat, lon, alt, focal_length)

    delta_x = gsd_w * pixel_x_dimension / 2
    delta_y = gsd_h * pixel_y_dimension / 2
    print(f'delta_x = {delta_x}, delta_y = {delta_y}')

    start_point = (lat, lon)

    # Distance in the horizontal (latitudinal) axis
    width_distance = distance(meters=delta_x)
    w_degrees = width_distance.destination(start_point, bearing=90).latitude - lat

    # Distance in the vertical (longitudinal) axis
    height_distance = distance(meters=delta_y)
    h_degrees = height_distance.destination(start_point, bearing=0).longitude - lon

    _, y = size
    angle_alpha = abs(angle % 360 - 180)
    print(f'angle_alpha = {angle_alpha}')
    print(f'angle_beta = {90 - angle_alpha}')
    print(f"y = {y}")
    rot_delta_x = (y * sin(radians(90 - angle_alpha)) / sin(radians(90))) * gsd_w
    rot_delta_y = (y * sin(radians(angle_alpha)) / sin(radians(90))) * gsd_h

    # Distance in the horizontal (latitudinal) axis
    width_distance = distance(meters=rot_delta_x)
    rot_w_degrees = width_distance.destination(start_point, bearing=90).latitude - lat

    # Distance in the vertical (longitudinal) axis
    height_distance = distance(meters=rot_delta_y)
    rot_h_degrees = height_distance.destination(start_point, bearing=0).longitude - lon

    print(f"rotated: {rot_delta_x}, {rot_delta_y}")

    return w_degrees, h_degrees, rot_w_degrees, rot_h_degrees


def rotate(img, df):
    lon1 = df['lon'].max()
    lat1 = df.loc[df['lon'] == lon1].iloc[0]['lat']
    lat2 = df['lat'].max()
    lon2 = df.loc[df['lat'] == lat2].iloc[0]['lon']

    angle = calculate_slope_angle(lat1, lon1, lat2, lon2)
    rotated = rotate_image(img, angle)
    return rotated, angle
