import folium
import math
from image_processing.rotation import calc_delta


def generate_map(images_df, directory, size_r):
    max_lat = images_df['lat'].max()
    min_lat = images_df['lat'].min()
    max_lon = images_df['lon'].max()
    min_lon = images_df['lon'].min()
    print(max_lat, min_lat, max_lon, min_lon)
    # gi.show_dots(images_df)

    center = [(max_lat + min_lat) / 2, (max_lon + min_lon) / 2]
    map = folium.Map(location=center,
                     zoom_start=18, width=1100, height=600, control_scale=True)

    img_dir = directory + '\\' + 'result_mosaic.png'

    x, y = size_r
    print(f"ratio {size_r}")
    delta = math.sqrt((max_lat - min_lat) ** 2 + (max_lon - min_lon) ** 2)
    bounds = [[min_lat - (y / x) * delta, min_lon - (x / y) * delta], [max_lat + (y / x) * delta, max_lon + (x / y) * delta]]
    # #bounds = [[min_lat - 0.15*delta - delta_x, min_lon - 0.15*delta - delta_y],
    # #          [max_lat + 0.15*delta + delta_x, max_lon + 0.15*delta + delta_x]]
    print(bounds)
    img_overlay = folium.raster_layers.ImageOverlay(name='image map',
                                                    image=img_dir,
                                                    bounds=bounds,
                                                    opacity=1,
                                                    interactive=True,
                                                    cross_origin=False,
                                                    zindex=1, )
    img_overlay.add_to(map)

    # for _, row in images_df.iterrows():
    #     folium.CircleMarker(location=(row['lat'], row['lon']), radius=2, popup=row['name'] + '/x' + str(row['lon'])
    #                                                                     + '/y' + str(row['lat'])).add_to(map)

    savepath = directory + '\\' + "map.html"
    map.save(savepath)


def make_map(images_df, angle, directory, size):
    max_lat = images_df['lat'].max()
    min_lat = images_df['lat'].min()
    max_lon = images_df['lon'].max()
    min_lon = images_df['lon'].min()
    # print(max_lat, min_lat, max_lon, min_lon)
    # gi.show_dots(images_df)

    center = [(max_lat + min_lat) / 2, (max_lon + min_lon) / 2]
    map = folium.Map(location=center,
                     zoom_start=18, width=1100, height=600, control_scale=True)

    img_dir = directory + '\\' + 'result_mosaic.png'

    delta_x, delta_y, rot_delta_x, rot_delta_y = calc_delta(directory, angle, size)
    print(f'delta_x: {delta_x}, delta_y: {delta_y}')
    print(f'rot_delta_x: {rot_delta_x}, rot_delta_y: {rot_delta_y}')

    bounds = [[min_lat - delta_x + rot_delta_x, min_lon - delta_y - rot_delta_y],
              [max_lat + delta_x - rot_delta_x, max_lon + delta_y + rot_delta_y]]
    print(f'bounds: {bounds}')
    img_overlay = folium.raster_layers.ImageOverlay(name='image map',
                                                    image=img_dir,
                                                    bounds=bounds,
                                                    opacity=1,
                                                    interactive=True,
                                                    cross_origin=False,
                                                    zindex=1, )
    img_overlay.add_to(map)

    # for _, row in images_df.iterrows():
    #     folium.CircleMarker(location=(row['lat'], row['lon']), radius=2, popup=row['name'] + '/x' + str(row['lon'])
    #                                                                     + '/y' + str(row['lat'])).add_to(map)

    savepath = directory + '\\' + "map.html"
    map.save(savepath)
