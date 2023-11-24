import numpy as np
import pandas as pd

def remove_duplicate_scenes(remove_from_set, keep_set):
    intersection = keep_set.intersection(remove_from_set)
    for element in intersection:
        remove_from_set.remove(element)

np.random.seed(100)
df_overview_path = "./data/classification_tags.csv"
df_overview = pd.read_csv(df_overview_path)

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

def load_scenes_by_categories():
    df_by_categories = []
    no_cat_scenes = df_overview
    for category in area_list:
        temp_df = df_overview[df_overview[category] == 1]
        no_cat_scenes = no_cat_scenes[df_overview[category] == 0]
        df_by_categories.append(temp_df)

    df_by_categories.append(no_cat_scenes)
    train_scenes = set()
    validation_scenes = set()
    test_scenes = set()
    for df in df_by_categories:
        num_scenes = len(df)
        test_size = round(0.2 * num_scenes)
        validation_size = round(0.1 * num_scenes)
        train_size = num_scenes - test_size - validation_size

        scenes = df["scene"].tolist()  # Assuming "scene" is a column in df
        np.random.shuffle(scenes)  # Shuffle the scenes in-place
        train_scenes = train_scenes.union(set(scenes[:train_size]))
        validation_scenes = validation_scenes.union(set(scenes[train_size:train_size + validation_size]))
        test_scenes = test_scenes.union(set(scenes[train_size + validation_size:train_size + validation_size + test_size]))
    #there are few scenes with more area types

    remove_duplicate_scenes(train_scenes, validation_scenes)
    remove_duplicate_scenes(test_scenes, train_scenes)
    remove_duplicate_scenes(validation_scenes, test_scenes)
    return list(train_scenes), list(validation_scenes), list(test_scenes)