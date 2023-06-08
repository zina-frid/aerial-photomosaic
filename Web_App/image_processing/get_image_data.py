import os
import pandas as pd
from exif import Image as eImage
from PIL import Image, ImageFile
import folium


# Converting coordinates to float format
def get_coordinates(image_object):
    lat_tuple = image_object.gps_latitude
    lon_tuple = image_object.gps_longitude
    lat_ref = image_object.gps_latitude_ref
    lon_ref = image_object.gps_longitude_ref
    lat_coord = lat_tuple[0] + lat_tuple[1] / 60 + lat_tuple[2] / 3600
    lon_coord = lon_tuple[0] + lon_tuple[1] / 60 + lon_tuple[2] / 3600
    if lat_ref == 'S':
        lat_coord = -lat_coord
    if lon_ref == 'W':
        lon_coord = -lon_coord
    return lat_coord, lon_coord


#  Creating a dataframe with metadata
def create_dataframe(images_dir):
    images_list = os.listdir(images_dir)
    images_df = pd.DataFrame(columns=['name', 'lat', 'lon', 'height', 'width'])
    images_dict = {}
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    for image in images_list:
        image_path = os.path.join(images_dir, image)
        image_read = open(image_path, 'rb')
        image_object = eImage(image_read)
        image_pic = Image.open(image_path)
        lat, lon = get_coordinates(image_object)
        height, width = image_object.pixel_y_dimension, image_object.pixel_x_dimension
        images_df.loc[images_df.size] = image, lat, lon, height, width
        # image_list.append([image, image_pic]);
        images_dict.update({image: image_pic})
        image_pic.close()

    return images_df, images_dict


# Marks on the map by image coordinates
def show_dots(images_df):
    meanLat = images_df.lat.mean()
    meanLon = images_df.lon.mean()
    map_flight = folium.Map(location=[meanLat, meanLon], zoom_start=20, control_scale=True)
    for _, row in images_df.iterrows():
        folium.CircleMarker(location=(row['lat'], row['lon']), radius=2, popup=row['name'] + '/x' + str(row['lon'])
                                                                        + '/y' + str(row['lat'])).add_to(map_flight)
    return map_flight

