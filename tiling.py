import torch

TILE_SIZE = 224
DOUBLE_TILE_SIZE = TILE_SIZE // 2

IMG_SIZE = 1022

TILE_PER_DIMENSION = IMG_SIZE // TILE_SIZE + 1
DOUBLE_TILE_PER_DIMENSION = 2 * (IMG_SIZE // TILE_SIZE) + 1

TILES_PER_IMAGE = pow(TILE_PER_DIMENSION, 2)
DOUBLE_TILES_PER_IMAGE = pow(DOUBLE_TILE_PER_DIMENSION, 2)


def get_tile_width_height_indexes(tile_index, double):
    tile_size = DOUBLE_TILE_SIZE if double else TILE_SIZE
    tile_per_dimension = DOUBLE_TILE_PER_DIMENSION if double else TILE_PER_DIMENSION
    tile_position_width = tile_index % tile_per_dimension
    tile_position_height = tile_index // tile_per_dimension
    tile_width_index = tile_position_width * tile_size
    tile_height_index = tile_position_height * tile_size
    if tile_position_width == tile_per_dimension - 1:
        tile_width_index = IMG_SIZE - TILE_SIZE
    if tile_position_height == tile_per_dimension - 1:
        tile_height_index = IMG_SIZE - TILE_SIZE
    return tile_width_index, tile_height_index


def get_tile(index, image, mask, double):
    tiles_per_image = DOUBLE_TILES_PER_IMAGE if double else TILES_PER_IMAGE
    tile_index = index % tiles_per_image
    tile_width_index, tile_height_index = get_tile_width_height_indexes(
        tile_index, double=double
    )
    image = image[
        tile_height_index : tile_height_index + TILE_SIZE,
        tile_width_index : tile_width_index + TILE_SIZE,
        :,
    ]
    mask = mask[
        tile_height_index : tile_height_index + TILE_SIZE,
        tile_width_index : tile_width_index + TILE_SIZE,
    ]
    return image, mask


def insert_tile_into_mask(tile_height_index, tile_width_index, insert_to_image, tile):
    insert_to_image[
        :,
        tile_height_index : tile_height_index + TILE_SIZE,
        tile_width_index : tile_width_index + TILE_SIZE,
    ] = (
        tile
        | insert_to_image[
            :,
            tile_height_index : tile_height_index + TILE_SIZE,
            tile_width_index : tile_width_index + TILE_SIZE,
        ]
    )
    return insert_to_image


def insert_tile_into_picture(
    tile_height_index, tile_width_index, insert_to_image, tile
):
    insert_to_image[
        :,
        tile_height_index : tile_height_index + TILE_SIZE,
        tile_width_index : tile_width_index + TILE_SIZE,
    ] = tile
    return insert_to_image


def get_masks_from_tiles(predicted_tiles, ground_truth_tiles):
    # C,H,W
    whole_ground_truth = torch.zeros((1, IMG_SIZE, IMG_SIZE), dtype=torch.int16)
    whole_predicted = whole_ground_truth.clone()
    for tile_index, predict_and_ground in enumerate(
        zip(predicted_tiles, ground_truth_tiles)
    ):
        predicted_tile = predict_and_ground[0]
        ground_truth_tile = predict_and_ground[1]
        tile_width_index, tile_height_index = get_tile_width_height_indexes(
            tile_index, double=False
        )
        whole_predicted = insert_tile_into_mask(
            tile_height_index, tile_width_index, whole_predicted, predicted_tile
        )
        whole_ground_truth = insert_tile_into_mask(
            tile_height_index, tile_width_index, whole_ground_truth, ground_truth_tile
        )
    return whole_predicted, whole_ground_truth


def get_picture_from_tile(image_tiles):
    whole_image = torch.zeros((4, IMG_SIZE, IMG_SIZE), dtype=torch.float32)
    for tile_index, tile in enumerate(image_tiles):
        tile_width_index, tile_height_index = get_tile_width_height_indexes(
            tile_index, double=False
        )
        whole_image = insert_tile_into_picture(
            tile_height_index, tile_width_index, whole_image, tile
        )
    return whole_image
