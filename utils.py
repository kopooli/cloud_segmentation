import dataset


def get_tile(index, image, mask):
    tile_index = index % dataset.TILES_PER_IMAGE
    tile_position_width = tile_index % dataset.TILE_PER_DIMENSION
    tile_position_height = tile_index // dataset.TILE_PER_DIMENSION
    tile_width_index = tile_position_width * dataset.TILE_SIZE
    tile_height_index = tile_position_height * dataset.TILE_SIZE
    if tile_position_width == dataset.TILE_PER_DIMENSION - 1:
        tile_width_index = dataset.IMG_SIZE - dataset.TILE_SIZE
    if tile_position_height == dataset.TILE_PER_DIMENSION - 1:
        tile_height_index = dataset.IMG_SIZE - dataset.TILE_SIZE
    image = image[
        tile_height_index : tile_height_index + dataset.TILE_SIZE,
        tile_width_index : tile_width_index + dataset.TILE_SIZE,
        :,
    ]
    mask = mask[
        tile_height_index : tile_height_index + dataset.TILE_SIZE,
        tile_width_index : tile_width_index + dataset.TILE_SIZE,
    ]
    return image, mask