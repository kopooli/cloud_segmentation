import numpy as np
import torch
import pandas as pd
from load_scenes_by_categories import load_scenes_by_categories
from dataset import CloudDataset
import dataset
from torchvision import transforms


def get_tile_width_height_indexes(tile_index):
    tile_position_width = tile_index % dataset.TILE_PER_DIMENSION
    tile_position_height = tile_index // dataset.TILE_PER_DIMENSION
    tile_width_index = tile_position_width * dataset.TILE_SIZE
    tile_height_index = tile_position_height * dataset.TILE_SIZE
    if tile_position_width == dataset.TILE_PER_DIMENSION - 1:
        tile_width_index = dataset.IMG_SIZE - dataset.TILE_SIZE
    if tile_position_height == dataset.TILE_PER_DIMENSION - 1:
        tile_height_index = dataset.IMG_SIZE - dataset.TILE_SIZE
    return tile_width_index, tile_height_index


def get_tile(index, image, mask):
    tile_index = index % dataset.TILES_PER_IMAGE
    tile_width_index, tile_height_index = get_tile_width_height_indexes(tile_index)
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


def get_picture_from_tiles(predicted_tiles, ground_truth_tiles):
    # C,H,W
    whole_ground_truth = torch.zeros(
        (1, dataset.IMG_SIZE, dataset.IMG_SIZE), dtype=torch.int16
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
            tile_height_index : tile_height_index + dataset.TILE_SIZE,
            tile_width_index : tile_width_index + dataset.TILE_SIZE,
        ] = (
            predicted_tile
            | whole_predicted[
                :,
                tile_height_index : tile_height_index + dataset.TILE_SIZE,
                tile_width_index : tile_width_index + dataset.TILE_SIZE,
            ]
        )
        whole_ground_truth[
            :,
            tile_height_index : tile_height_index + dataset.TILE_SIZE,
            tile_width_index : tile_width_index + dataset.TILE_SIZE,
        ] = (
            ground_truth_tile
            | whole_ground_truth[
                :,
                tile_height_index : tile_height_index + dataset.TILE_SIZE,
                tile_width_index : tile_width_index + dataset.TILE_SIZE,
            ]
        )
    return whole_predicted, whole_ground_truth
