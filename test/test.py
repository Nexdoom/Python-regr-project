#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import regression as rg
import data_processing as dproc

# ========================================================================

EXP_DATA_DIR = r"Data\exp"
CALC_DATA_DIR = r"Data\calc"
OUT_DIR = r"Result"

# ========================================================================

#dataset_exp = {"name": "Empty",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": (),
#                "segments_config": ("x**0.2"),
#                "output_regr_and_predict": True}
#
#dataset_calc = {"name": "Empty",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": (),
#                "segments_config": ("x**0.2"),
#                "output_regr_and_predict": True}


#dataset_exp = {"name": "test_one_segm",
#               "dir": EXP_DATA_DIR,
#               "exclude_filters": (),
#               "segments_config": ("x**2"),
#               "prediction_points_src": "exp",
#               "output_regr_and_predict": False}
#
#dataset_calc = {"name": "test_one_segm",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": (),
#                "segments_config": ("x+x**2+x**3+x**4+x**5+x**6"),
#                "prediction_points_src": "exp",
#                "output_regr_and_predict": False}


#dataset_exp = {"name": "test_two_segm",
#               "dir": EXP_DATA_DIR,
#               "exclude_filters": (),
#               "segments_config": ("x**-1.7", 1023.0, "x"),
#               "prediction_points_src": "exp",
#               "output_regr_and_predict": True}
#
#dataset_calc = {"name": "test_two_segm",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": ((2000.0, 3000.0),),
#                "segments_config": ("x**0.2", 1023.0, "x**-1"),
#                "prediction_points_src": "calc",
#                "output_regr_and_predict": True}


#dataset_exp = {"name": "test_three_segm",
#               "dir": EXP_DATA_DIR,
#               "exclude_filters": ((0.8, -1),),
#               "segments_config": ("x**-0.4", 0.3, "x", 0.545, "x**2"),
#               "prediction_points_src": "exp",
#               "output_regr_and_predict": True}
#
#dataset_calc = {"name": "test_three_segm",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": (),
#                "segments_config": ("x+x**2+x**3+x**4+x**5+x**6"),
#                "prediction_points_src": "exp",
#                "output_regr_and_predict": True}

# ========================================================================

def plot_segments(datasets):
    for dataset in datasets:
        full_data = dataset["data"]
        segments = dataset["segments"]

        _, ax = plt.subplots(figsize=(12, 8))
        ax.plot(full_data["x"], full_data["y"], 'o', label="Data")

        for segment in segments:
            N_PREDICT_POINTS = 100
            prediction_points = np.linspace(segment["bnd_left"],
                                            segment["bnd_right"],
                                            N_PREDICT_POINTS)

            prediction_points = pd.Series(prediction_points, name="x")

            regr_result = rg.ols_fit(segment["regr_func"], segment["segment_data"])
            prediction = rg.get_prediction(regr_result, prediction_points, verbose=False)

            label = "OLS prediction: ({}-{})".format(round(segment["bnd_left"], 1),
                                                     round(segment["bnd_right"], 1))

            ax.plot(prediction["x"], prediction["mean"], '-', label=label)
            ax.legend(loc="best")


if __name__ == "__main__":
    datasets = (dataset_exp, dataset_calc)

    datasets = dproc.add_data_and_segments(datasets)

    datasets = rg.add_prediction_result(datasets)

    plot_segments(datasets)

    rg.save_prediction(OUT_DIR, datasets)
