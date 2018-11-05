import os
import pandas as pd
from enum import Enum


Direction = Enum("Direction", "left right")


class InputDataError(Exception):
    pass


def load_data_csv(data_dir, data_name, sep=","):
	data_file_name = data_name + ".csv"
	data_file_path = os.path.join(data_dir, data_file_name)
	data = pd.read_csv(data_file_path, names=["x","y"], sep=sep)
	data.sort_values("x").reset_index(drop=True, inplace=True)
	return data


def save_dataframe_csv(dir_name, dataframe_name, dataframe, header=True, sep=","):
	file_path = os.path.join(dir_name, dataframe_name + ".csv")
	dataframe.to_csv(file_path, index=False, header=header, sep=sep)


def get_nearest_point_index(data, x_value, direction):
    if direction is Direction.left:
        point_index = max(data[x_value >= data["x"]].index.tolist())
    elif direction is Direction.right:
        point_index = min(data[x_value <= data["x"]].index.tolist())

    return point_index


def get_point_indexes_to_drop(data, filters):

    point_indexes = []
    for (initial_left_bnd, initial_right_bnd) in filters:
        initial_filter_desc = "{!s}".format((initial_left_bnd, initial_right_bnd))

        try:
            if isinstance(initial_left_bnd, float):
                left_bnd = get_nearest_point_index(data, initial_left_bnd, Direction.right)
            else:
                left_bnd = initial_left_bnd

            if isinstance(initial_right_bnd, float):
                right_bnd = get_nearest_point_index(data, initial_right_bnd, Direction.left)
            else:
                right_bnd = initial_right_bnd
        except(ValueError):
            raise InputDataError("No points to delete in specified range: {}"
                              .format(initial_filter_desc))

        if (left_bnd > right_bnd) and (right_bnd != -1):
            raise InputDataError("Boundaries are not ascending in specified range: {!s}"
                              .format(initial_filter_desc))

        if left_bnd == -1:
            left_point_index = data.index.min()
        else:
            left_point_index = left_bnd

        if right_bnd == -1:
            right_point_index = data.index.max()
        else:    
            right_point_index = right_bnd

        if (left_point_index == 0) or (right_point_index == 0):
            raise InputDataError("Zero boundary error. Boundary=0 in specified range: {}"
                              .format(initial_filter_desc))

        if (left_point_index < data.index.min()) or (right_point_index > data.index.max()):
            raise InputDataError("Filter {0!s} exceeds data range. Data has {1!s} points"
                              .format(initial_filter_desc, data.index.max()))

        point_indexes.extend(range(left_point_index, right_point_index+1))

    point_indexes = list(set(point_indexes))

    return point_indexes


def get_filtred_data(dataset):
    data = dataset["data"]

    if "exclude_filters" not in dataset:
        return data

    filters = dataset["exclude_filters"]

    data.index += 1

    point_indexes_to_drop = get_point_indexes_to_drop(data, filters)
    data.drop(point_indexes_to_drop, inplace=True)

    data.reset_index(drop=True, inplace=True)

    return data


def add_data_to_segments(segments, full_data):
    for segment in segments:
        segment_data = full_data[(segment["bnd_left"] <= full_data["x"]) &
                                 (full_data["x"] <= segment["bnd_right"])]
        segment_data.reset_index(drop=True, inplace=True)

        if segment_data.empty == True:
            raise InputDataError("Segment starting at {!s} is empty".format(segment["bnd_left"]))

        segment["segment_data"] = segment_data

    return segments


def create_segments(full_data, segments_config):
    if isinstance(segments_config, str):
        segments_config = (segments_config,)

    full_data_bnd_left = full_data["x"].min()
    full_data_bnd_right = full_data["x"].max()

    x_values = [full_data_bnd_left] + \
               [float(val) for val in segments_config if not isinstance(val, str)] + \
               [full_data_bnd_right]

    regr_functions = [val for val in segments_config if isinstance(val, str)]

    segments = []
    for (index, regr_func) in enumerate(regr_functions):
        segment = {"regr_func": regr_func,
                   "bnd_left": x_values[index],
                   "bnd_right": x_values[index+1]}
        segments.append(segment)

    segments = add_data_to_segments(segments, full_data)

    return segments


def add_prediction_point_to_datasets(datasets):
    dataset_exp, dataset_calc = datasets[:]

    if dataset_exp["prediction_points_src"].lower() == "exp":
        dataset_exp["prediction_points"] = dataset_exp["data"]["x"]
    elif dataset_exp["prediction_points_src"].lower() == "calc":
        dataset_exp["prediction_points"] = dataset_calc["data"]["x"]
    else:
        raise InputDataError("Invalid value in \"prediction_points_src\"")

    if dataset_calc["prediction_points_src"].lower() == "calc":
        dataset_calc["prediction_points"] = dataset_calc["data"]["x"]
    elif dataset_calc["prediction_points_src"].lower() == "exp":
        dataset_calc["prediction_points"] = dataset_exp["data"]["x"]
    else:
        raise InputDataError("Invalid value in \"prediction_points_src\"")

    return datasets


def add_prediction_point_to_segments(segments, prediction_points):
    for i, segment in enumerate(segments):
        if len(segments) == 1:
            segment_prediction_points = prediction_points
        elif i == 0:
            segment_prediction_points = prediction_points[prediction_points <= segment["bnd_right"]]
        elif i == len(segments)-1:
            segment_prediction_points = prediction_points[segment["bnd_left"] <= prediction_points]
        else:
            segment_prediction_points = prediction_points[(segment["bnd_left"] <= prediction_points) &
                                                          (prediction_points <= segment["bnd_right"])]
        
        segment_prediction_points.reset_index(drop=True, inplace=True)
        
        if segment_prediction_points.empty == True:
            raise InputDataError("Segment starting at {!s} is empty".format(segment["bnd_left"]))
        
        segment["segment_prediction_points"] = segment_prediction_points

    return segments


def add_data_and_segments(datasets):
    for dataset in datasets:
        dataset["data"] = load_data_csv(dataset["dir"], dataset["name"], sep=";")

        if dataset["data"].empty == True:
            raise InputDataError("File {!s} is empty".format(dataset["name"] + ".csv"))

        dataset["data"] = dataset["data"].drop_duplicates(subset=['x', 'y'], keep="first")
        dataset["data"].reset_index(drop=True, inplace=True)

        dataset["data"] = get_filtred_data(dataset)

    datasets = add_prediction_point_to_datasets(datasets)

    for dataset in datasets:
        dataset["segments"] = create_segments(dataset["data"], dataset["segments_config"])
        dataset["segments"] = add_prediction_point_to_segments(dataset["segments"], dataset["prediction_points"])

    return datasets