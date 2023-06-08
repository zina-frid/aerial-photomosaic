import webbrowser
from image_processing.get_image_data import show_dots, create_dataframe

dir = ""
save_path = ""

images_df, images_dict = create_dataframe(dir)
map_ = show_dots(images_df)
map_.save(save_path)
webbrowser.open(save_path)
