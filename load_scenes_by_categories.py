import numpy as np
import pandas as pd
np.random.seed(100)

def remove_duplicate_scenes(first_set, second_set):
    intersection = list(first_set.intersection(second_set))
    intersection.sort()
    np.random.shuffle(intersection)
    one_half_index = len(intersection) // 2
    for element in intersection[:one_half_index]:
        first_set.remove(element)
    for element in intersection[one_half_index:]:
        second_set.remove(element)


df_overview_path = "./data/classification_tags.csv"
df_overview = pd.read_csv(df_overview_path)
df_overview.sort_values(by=["scene"])

area_list = [
    "forest/jungle",
    "snow/ice",
    "agricultural",
    "urban/developed",
    "coastal",
    "hills/mountains",
    "desert/barren",
    "shrublands/plains",
    "wetland/bog/marsh",
    "open_water",
    "enclosed_water",
]

"""area_list = [
    "cumulus",
    "cumulonimbus",
    "altocumulus/stratocumulus",
    "cirrus",
    "haze/fog",
    "ice_clouds",
    "contrails",
]"""


def load_scenes_by_categories():
    df_by_categories = []
    no_catg_scenes = df_overview
    for category in area_list:
        temp_df = df_overview[df_overview[category] == 1]
        no_catg_scenes = no_catg_scenes[df_overview[category] == 0]
        df_by_categories.append(temp_df)

    df_by_categories.append(no_catg_scenes)
    train_scenes = set()
    validation_scenes = set()
    test_scenes = set()
    for df in df_by_categories:
        num_scenes = len(df)
        test_size = round(0.2 * num_scenes)
        validation_size = round(0.15 * num_scenes)
        train_size = num_scenes - test_size - validation_size

        scenes = df["scene"].tolist()  # Assuming "scene" is a column in df
        np.random.shuffle(scenes)  # Shuffle the scenes in-place

        train_scenes = train_scenes.union(set(scenes[:train_size]))
        validation_scenes = validation_scenes.union(
            set(scenes[train_size : train_size + validation_size])
        )
        test_scenes = test_scenes.union(set(scenes[train_size+ validation_size:]))
    # there are few scenes with more area types
    remove_duplicate_scenes(train_scenes, validation_scenes)
    remove_duplicate_scenes(test_scenes, train_scenes)
    remove_duplicate_scenes(validation_scenes, test_scenes)
    train_scenes = list(train_scenes)
    validation_scenes = list(validation_scenes)
    test_scenes = list(test_scenes)
    train_scenes.sort()
    validation_scenes.sort()
    test_scenes.sort()
    return train_scenes, validation_scenes, test_scenes
