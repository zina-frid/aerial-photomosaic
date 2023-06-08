from PIL import Image
import os


def create_image_tiles(directory):
    output_folder = directory + '\\' + "tiles"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image = Image.open(directory + '\\' + "result_mosaic.png")
    tile_size = (256, 256)
    current_size = image.size
    level = 0

    while current_size[0] >= 256 and current_size[1] >= 256:
        level_folder = os.path.join(output_folder, f'level_{level}')
        os.makedirs(level_folder, exist_ok=True)

        if current_size[0] % tile_size[0] != 0 or current_size[1] % tile_size[1] != 0:
            # Add a transparent background for the missing pixels
            width_diff = tile_size[0] - current_size[0] % tile_size[0]
            height_diff = tile_size[1] - current_size[1] % tile_size[1]
            background = Image.new('RGBA', (current_size[0] + width_diff, current_size[1] + height_diff), (0, 0, 0, 0))
            background.paste(image, (0, 0))
            image = background
            current_size = image.size

        tiles = []
        for left in range(0, current_size[0], tile_size[0]):
            for top in range(0, current_size[1], tile_size[1]):
                right = min(left + tile_size[0], current_size[0])
                bottom = min(top + tile_size[1], current_size[1])
                tile = image.crop((left, top, right, bottom))
                tiles.append(tile)

        for i, tile in enumerate(tiles):
            x = i % (current_size[0] // tile_size[0])
            y = i // (current_size[0] // tile_size[0])
            save_path = os.path.join(level_folder, f'tile_{x}_{y}.png')
            tile.save(save_path)

        # Reduce image size for next iteration
        image = image.resize((current_size[0] // 2, current_size[1] // 2))
        current_size = image.size
        level += 1

    # Save the last tile if its size is less than 256x256
    if current_size[0] < 256 or current_size[1] < 256:
        level_folder = os.path.join(output_folder, f'level_{level}')
        os.makedirs(level_folder, exist_ok=True)
        save_path = os.path.join(level_folder, 'tile_0_0.png')
        image.save(save_path)
