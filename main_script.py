#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import regression as rg
import data_processing as dproc

# ========================================================================

EXP_DATA_DIR = r"..\!Data\Exp"
CALC_DATA_DIR = r"..\!Data\Calc"
OUT_DIR = r"..\Out"

# ========================================================================

#EXP_NAME = "tg_tanaka_exp_f"
#REGR_FUNC_CALC = "x"
#REGR_FUNC_EXP = "x**0.5+x**-1"

#EXP_NAME = "zahlebivanie"
#REGR_FUNC_CALC = "x"
#REGR_FUNC_EXP = "x"

#EXP_NAME = "shah_jackson_2.1_down"
#REGR_FUNC_CALC = "x**-0.8+x**0.4"
#REGR_FUNC_EXP = "x**-0.4"

#EXP_NAME = "shah_jackson_2.1_up"
#REGR_FUNC_CALC = "x**-1+x**1+x**2+x**3+x**4+x**5+x**6"
#REGR_FUNC_EXP = "x**1+x**-1"

#EXP_NAME = "shah_jackson_7.1_down"
#REGR_FUNC_CALC = "x**-0.5+x**0.5"
#REGR_FUNC_EXP = "x**-0.2+x**0.5"

#EXP_NAME = "shah_jackson_7.1_up"
#REGR_FUNC_CALC = "x**2+x"
#REGR_FUNC_EXP = "x**1+x**-1+x**-2"

#EXP_NAME = "tanrikut_1.2.1_tw"
#REGR_FUNC_CALC = "x+x**-3"
#REGR_FUNC_EXP = "x+x**-3"

#EXP_NAME = "akimoto_exper001"
#REGR_FUNC_CALC = "x"
#REGR_FUNC_EXP = "x"

#EXP_NAME = "akimoto_exper002"
#REGR_FUNC_CALC = "x"
#REGR_FUNC_EXP = "x"

#EXP_NAME = "akimoto_exper003"
#REGR_FUNC_CALC = "x"
#REGR_FUNC_EXP = "x"

#EXP_NAME = "akimoto_exper004"
#REGR_FUNC_CALC = "x"
#REGR_FUNC_EXP = "x"


#dataset_calc = {"name": "Empty",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": ((1, 5),),
#                "segments_config": ("x**0.2", 1020.0, "x**-1"),
#                "output_regr_and_predict": True}


#dataset_exp = {"name": "karoutas001_t1",
#               "dir": EXP_DATA_DIR,
#               "exclude_filters": (),
#               "segments_config": ("x**-1.7", 1023.0, "x"),
#               "prediction_points_src": "exp",
#               "output_regr_and_predict": True}
#
#dataset_calc = {"name": "karoutas001_t1",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": ((2000.0, 3000.0),),
#                "segments_config": ("x**0.2", 1023.0, "x**-1"),
#                "prediction_points_src": "calc",
#                "output_regr_and_predict": True}


#dataset_exp = { "name": "karoutas001_t2",
#                "dir": EXP_DATA_DIR,
#                "exclude_filters": ((2000.0, 3000.0),),
#                "segments_config": ("x**-1", 1007.0, "x"),
#                "prediction_points_src": "exp",
#                "output_regr_and_predict": True}
#
#dataset_calc = { "name": "karoutas001_t2",
#                 "dir": CALC_DATA_DIR,
#                 "exclude_filters": ((2000.0, 3000.0),),
#                 "segments_config": ("x**0.3", 1020.0, "x**-1"),
#                 "prediction_points_src": "calc",
#                 "output_regr_and_predict": True}


#dataset_exp = {"name": "karoutas001_t3",
#               "dir": EXP_DATA_DIR,
#               "exclude_filters": ((21, -1),),
#               "segments_config": ("x**-0.8", 1003.0, "x"),
#               "prediction_points_src": "exp",
#               "output_regr_and_predict": True}
#
#dataset_calc = {"name": "karoutas001_t3",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": ((21, -1),),
#                "segments_config": ("x**0.4", 1003.0, "x**-1"),
#                "prediction_points_src": "calc",
#                "output_regr_and_predict": True}


#dataset_exp = {"name": "karoutas001_t4",
#               "dir": EXP_DATA_DIR,
#               "exclude_filters": ((21, 21),),
#               "segments_config": ("x**-0.8", 943.0, "x"),
#               "prediction_points_src": "exp",
#               "output_regr_and_predict": True}
#
#dataset_calc = {"name": "karoutas001_t4",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": ((21, 21),),
#                "segments_config": ("x**0.9", 943.0, "x**-1"),
#                "prediction_points_src": "calc",
#                "output_regr_and_predict": True}


#dataset_exp = {"name": "karoutas002_t1",
#               "dir": EXP_DATA_DIR,
#               "exclude_filters": ((1900.0, -1),),
#               "segments_config": ("x**-0.4", 940.0, "x"),
#               "prediction_points_src": "exp",
#               "output_regr_and_predict": True}
#
#dataset_calc = {"name": "karoutas002_t1",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": ((1900.0, -1),),
#                "segments_config": ("x**0.4", 940.0, "x**-1"),
#                "prediction_points_src": "calc",
#                "output_regr_and_predict": True}


#dataset_exp = {"name": "karoutas002_t2",
#               "dir": EXP_DATA_DIR,
#               "exclude_filters": ((15, 15),),
#               "segments_config": ("x**-0.3", 954.0, "x"),
#               "prediction_points_src": "exp",
#               "output_regr_and_predict": True}
#
#dataset_calc = {"name": "karoutas002_t2",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": ((15, 15),),
#                "segments_config": ("x**0.4", 954.0, "x**-1"),
#                "prediction_points_src": "calc",
#                "output_regr_and_predict": True}


#dataset_exp = {"name": "karoutas002_t3",
#               "dir": EXP_DATA_DIR,
#               "exclude_filters": ((15, 15),),
#               "segments_config": ("x**-1.1", 1045.0, "x"),
#               "prediction_points_src": "exp",
#               "output_regr_and_predict": True}
#
#dataset_calc = {"name": "karoutas002_t3",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": ((15, 15),),
#                "segments_config": ("x**0.1", 1045.0, "x**-0.8"),
#                "prediction_points_src": "calc",
#                "output_regr_and_predict": True}


#dataset_exp = {"name": "karoutas002_t4",
#               "dir": EXP_DATA_DIR,
#               "exclude_filters": ((15, 15),),
#               "segments_config": ("x**-1.1", 950.0, "x"),
#               "prediction_points_src": "exp",
#               "output_regr_and_predict": True}
#
#dataset_calc = {"name": "karoutas002_t4",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": ((15, 15),),
#                "segments_config": ("x**0.8", 950.0, "x**-1"),
#                "prediction_points_src": "calc",
#                "output_regr_and_predict": True}


#dataset_exp = {"name": "bosio&imset_exper001",
#               "dir": EXP_DATA_DIR,
#               "exclude_filters": (),
#               "segments_config": ("x**2"),
#               "prediction_points_src": "exp",
#               "output_regr_and_predict": True}
#
#dataset_calc = {"name": "bosio&imset_exper001",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": (),
#                "segments_config": ("x+x**2+x**3+x**4+x**5+x**6"),
#                "prediction_points_src": "exp",
#                "output_regr_and_predict": True}


#dataset_exp = {"name": "wang_exper001",
#               "dir": EXP_DATA_DIR,
#               "exclude_filters": (),
#               "segments_config": ("x**2+x**3"),
#               "prediction_points_src": "exp",
#               "output_regr_and_predict": True}
#
#dataset_calc = {"name": "wang_exper001",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": (),
#                "segments_config": ("x+x**2+x**3+x**4+x**5"),
#                "prediction_points_src": "exp",
#                "output_regr_and_predict": True}


#dataset_exp = {"name": "wang_exper002",
#               "dir": EXP_DATA_DIR,
#               "exclude_filters": (),
#               "segments_config": ("x**2+x**3"),
#               "prediction_points_src": "exp",
#               "output_regr_and_predict": True}
#
#dataset_calc = {"name": "wang_exper002",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": (),
#                "segments_config": ("x+x**2+x**3+x**4+x**5"),
#                "prediction_points_src": "exp",
#                "output_regr_and_predict": True}


#dataset_exp = {"name": "wang_exper003",
#               "dir": EXP_DATA_DIR,
#               "exclude_filters": (),
#               "segments_config": ("x**0.9"),
#               "prediction_points_src": "exp",
#               "output_regr_and_predict": True}
#
#dataset_calc = {"name": "wang_exper003",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": (),
#                "segments_config": ("x+x**2+x**3+x**4+x**5"),
#                "prediction_points_src": "exp",
#                "output_regr_and_predict": True}


#dataset_exp = {"name": "wang_exper004",
#               "dir": EXP_DATA_DIR,
#               "exclude_filters": (),
#               "segments_config": ("x**1"),
#               "prediction_points_src": "exp",
#               "output_regr_and_predict": True}
#
#dataset_calc = {"name": "wang_exper004",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": (),
#                "segments_config": ("x+x**2+x**3+x**4+x**5"),
#                "prediction_points_src": "exp",
#                "output_regr_and_predict": True}


#dataset_exp = {"name": "wang_exper005",
#               "dir": EXP_DATA_DIR,
#               "exclude_filters": (),
#               "segments_config": ("x**0.7"),
#               "prediction_points_src": "exp",
#               "output_regr_and_predict": True}
#
#dataset_calc = {"name": "wang_exper005",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": (),
#                "segments_config": ("x+x**2+x**3+x**4+x**5"),
#                "prediction_points_src": "exp",
#                "output_regr_and_predict": True}


#dataset_exp = {"name": "adiabat_o_exper001",
#               "dir": EXP_DATA_DIR,
#               "exclude_filters": (),
#               "segments_config": ("x+x**2+x**3"),
#               "prediction_points_src": "exp",
#               "output_regr_and_predict": True}
#
#dataset_calc = {"name": "adiabat_o_exper001",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": (),
#                "segments_config": ("x+x**2+x**3+x**4+x**5"),
#                "prediction_points_src": "exp",
#                "output_regr_and_predict": True}


#dataset_exp = {"name": "adiabat_o_exper002",
#               "dir": EXP_DATA_DIR,
#               "exclude_filters": (),
#               "segments_config": ("x**2"),
#               "prediction_points_src": "exp",
#               "output_regr_and_predict": True}
#
#dataset_calc = {"name": "adiabat_o_exper002",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": ((-1, 0.5),),
#                "segments_config": ("x+x**2+x**3+x**4+x**5+x**6"),
#                "prediction_points_src": "exp",
#                "output_regr_and_predict": True}


#dataset_exp = {"name": "adiabat_o_exper003",
#               "dir": EXP_DATA_DIR,
#               "exclude_filters": (),
#               "segments_config": ("x+x**2+x**3"),
#               "prediction_points_src": "exp",
#               "output_regr_and_predict": True}
#
#dataset_calc = {"name": "adiabat_o_exper003",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": (),
#                "segments_config": ("x+x**2+x**3+x**4+x**5+x**6"),
#                "prediction_points_src": "exp",
#                "output_regr_and_predict": True}


#dataset_exp = {"name": "adiabat_o_exper004",
#               "dir": EXP_DATA_DIR,
#               "exclude_filters": (),
#               "segments_config": ("x+x**2+x**3"),
#               "prediction_points_src": "exp",
#               "output_regr_and_predict": True}
#
#dataset_calc = {"name": "adiabat_o_exper004",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": ((-1, 0.2),),
#                "segments_config": ("x+x**2+x**3+x**4+x**5+x**6"),
#                "prediction_points_src": "exp",
#                "output_regr_and_predict": True}


#dataset_exp = {"name": "adiabat_o_exper005",
#               "dir": EXP_DATA_DIR,
#               "exclude_filters": (),
#               "segments_config": ("x+x**2+x**3"),
#               "prediction_points_src": "exp",
#               "output_regr_and_predict": True}
#
#dataset_calc = {"name": "adiabat_o_exper005",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": (),
#                "segments_config": ("x+x**2+x**3+x**4+x**5+x**6"),
#                "prediction_points_src": "exp",
#                "output_regr_and_predict": True}


#dataset_exp = {"name": "adiabat_p_exper002",
#               "dir": EXP_DATA_DIR,
#               "exclude_filters": (),
#               "segments_config": ("x+x**2+x**3"),
#               "prediction_points_src": "exp",
#               "output_regr_and_predict": True}
#
#dataset_calc = {"name": "adiabat_p_exper002",
#                "dir": CALC_DATA_DIR,
#                "exclude_filters": (),
#                "segments_config": ("x+x**2+x**3+x**4+x**5+x**6"),
#                "prediction_points_src": "exp",
#                "output_regr_and_predict": True}


#dataset_exp = {"name": "tarasova_exper001",
#               "dir": EXP_DATA_DIR,
#               "exclude_filters": (),
#               "segments_config": ("x**-1+x+x**2"),
#               "prediction_points_src": "exp",
#               "output_regr_and_predict": True}
#
#dataset_calc = {"name": "tarasova_exper001",
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
