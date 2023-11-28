import torch

TILE_SIZE = 224
IMG_SIZE = 1022
TILE_PER_DIMENSION = IMG_SIZE // TILE_SIZE + 1
TILES_PER_IMAGE = pow(TILE_PER_DIMENSION, 2)

def get_tile_width_height_indexes(tile_index):
    tile_position_width = tile_index % TILE_PER_DIMENSION
    tile_position_height = tile_index // TILE_PER_DIMENSION
    tile_width_index = tile_position_width * TILE_SIZE
    tile_height_index = tile_position_height * TILE_SIZE
    if tile_position_width == TILE_PER_DIMENSION - 1:
        tile_width_index = IMG_SIZE - TILE_SIZE
    if tile_position_height == TILE_PER_DIMENSION - 1:
        tile_height_index = IMG_SIZE - TILE_SIZE
    return tile_width_index, tile_height_index


def get_tile(index, image, mask):
    tile_index = index % TILES_PER_IMAGE
    tile_width_index, tile_height_index = get_tile_width_height_indexes(tile_index)
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


def get_picture_from_tiles(predicted_tiles, ground_truth_tiles):
    # C,H,W
    whole_ground_truth = torch.zeros(
        (1, IMG_SIZE, IMG_SIZE), dtype=torch.int16
    )
    whole_predicted = whole_ground_truth.clone()
    for tile_index, predict_and_ground in enumerate(
        zip(predicted_tiles, ground_truth_tiles)
    ):
        predicted_tile = predict_and_ground[0]
        ground_truth_tile = predict_and_ground[1]
        tile_width_index, tile_height_index = get_tile_width_height_indexes(tile_index)
        whole_predicted[
            :,
            tile_height_index : tile_height_index + TILE_SIZE,
            tile_width_index : tile_width_index + TILE_SIZE,
        ] = (
            predicted_tile
            | whole_predicted[
                :,
                tile_height_index : tile_height_index + TILE_SIZE,
                tile_width_index : tile_width_index + TILE_SIZE,
            ]
        )
        whole_ground_truth[
            :,
            tile_height_index : tile_height_index + TILE_SIZE,
            tile_width_index : tile_width_index + TILE_SIZE,
        ] = (
            ground_truth_tile
            | whole_ground_truth[
                :,
                tile_height_index : tile_height_index + TILE_SIZE,
                tile_width_index : tile_width_index + TILE_SIZE,
            ]
        )
    return whole_predicted, whole_ground_truth
