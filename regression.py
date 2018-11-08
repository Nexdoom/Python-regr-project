import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import sympy as sp
import sympy.parsing.sympy_parser as sp_parser

import data_processing as dproc


N_PREDICT_POINTS = 100
FLOAT_REGEX = r"[+-]?([0-9]*[.])?[0-9]+"


class InputDataError(Exception):
    pass


def ols_fit(regr_func, data):
	full_regex = r"(\w+)\*\*(" + FLOAT_REGEX + ")"
	regr_func = re.sub(full_regex, r"pow(\1, \2)", regr_func)
	regr_func = "y ~ " + regr_func

	regr_result = smf.ols(formula=regr_func, data=data).fit()
	return regr_result


def print_regress_stats(data_name, regr_result):
	print("==============================================================================")
	print("Name:", data_name)
	print("Regression function:", regr_result.model.formula)
	print("==============================================================================")

	print(regr_result.summary())


def plot_regress_stats(regr_result):
	fig = plt.figure(figsize=(12, 8))
	sm.graphics.plot_regress_exog(regr_result, 1, fig=fig)

	fig = plt.figure(figsize=(12, 8))
	sm.graphics.plot_partregress_grid(regr_result, fig=fig)


def plot_regress_results(data, regr_result):
	x = np.linspace(data["x"].min(), data["x"].max(), N_PREDICT_POINTS)
	x = pd.Series(x, name="x")
	predictions = get_prediction(regr_result, x)

	fig, ax = plt.subplots(figsize=(12, 8))
	ax.plot(data["x"], data["y"], 'o', label="Data")
	ax.plot(predictions["x"], predictions["mean"], 'b-', label="OLS prediction")
	ax.fill_between(predictions["x"],
	                predictions["mean_ci_lower"],
	                predictions["mean_ci_upper"],
	                alpha=0.15,
	                label="Mean CI")
	ax.fill_between(predictions["x"],
	                predictions["mean_se_lower"],
	                predictions["mean_se_upper"],
	                alpha=0.5,
	                label="Mean SE")
	ax.legend(loc="best")


def output_regress_results(data_name, data, regr_result):
	print_regress_stats(data_name, regr_result)
	plot_regress_stats(regr_result)
	plot_regress_results(data, regr_result)


def get_result_function(regr_result):
	regr_func = ""
	for index in reversed(regr_result.params.index):
		if index == "Intercept":
			continue

		coef = "{0:+}".format(regr_result.params[index])

		full_regex = r"pow\(x, (" + FLOAT_REGEX + ")\)"
		index = re.sub(full_regex, r"x**\1", index)

		regr_func = regr_func + coef + "*" + index
	return regr_func


def get_function_derivative(function, x_symbol, verbose=False):
	y_func = sp_parser.parse_expr(function)
	x_symbol = sp.Symbol(x_symbol)
	y_diff = y_func.diff(x_symbol)

	if verbose:
		print("")
		print("==============================================================================")
		print("                           Derivatives calculation")
		print("==============================================================================")
		print("Original function:", function)
		print("Parsed function:", y_func)
		print("Derivative function:", y_diff)
		print("==============================================================================")
		print("")

	func_deriv = sp.lambdify(x_symbol, y_diff)

	return func_deriv


def calc_derivatives(function, x, x_symbol, verbose=False):
	func_deriv = get_function_derivative(function, x_symbol=x_symbol, verbose=verbose)
	derivatives = func_deriv(x)

	# const "derivatives" returned from "func_deriv" if derivative is const
	if not isinstance(derivatives, pd.Series):
		derivatives = pd.Series(np.repeat(derivatives, x.size))

	derivatives.name = "deriv"

	return derivatives


def get_prediction(regr_result, x, verbose=False):
	predictions = regr_result.get_prediction(pd.DataFrame(x))
	summary = predictions.summary_frame(alpha=0.05)

	mean_se_upper = pd.Series(summary["mean"] + summary["mean_se"], name="mean_se_upper")
	mean_se_lower = pd.Series(summary["mean"] - summary["mean_se"], name="mean_se_lower")

	mean_ci = pd.Series.abs(summary["mean"] - summary["mean_ci_lower"])
	mean_ci.name = "mean_ci"

	result_func = get_result_function(regr_result)
	derivatives = calc_derivatives(result_func, x, 'x', verbose=verbose)

	result = pd.concat([x,
	                    summary["mean"],
	                    derivatives,
	                    mean_ci,
	                    summary["mean_ci_lower"],
	                    summary["mean_ci_upper"],
	                    summary["mean_se"],
	                    mean_se_lower,
	                    mean_se_upper],
	                   axis=1)

	return result


def plot_prediction_data(predictions):
	fig, ax = plt.subplots(figsize=(12, 8))
	ax.errorbar(predictions["x"], predictions["mean"], yerr=predictions["mean_ci"],
	            fmt='o', ecolor='black', mfc='black', mec='black', capsize=2, ms=4, mew=1)
	ax.legend(loc="best")


def add_prediction_result(datasets):
    for dataset in datasets:
        for segment in dataset["segments"]:
            regr_result = ols_fit(segment["regr_func"], segment["segment_data"])
            prediction = get_prediction(regr_result, segment["segment_prediction_points"], verbose=True)

            output_result=dataset.get("output_regr_and_predict")
            if output_result == True:
                output_regress_results(dataset["name"], segment["segment_data"], regr_result)
                plot_prediction_data(prediction)
            elif output_result == False:
                pass
            else:
                print("Warning! Option: \"output_regr_and_predict\" not entered, default=True")
                output_regress_results(dataset["name"], segment["segment_data"], regr_result)
                plot_prediction_data(prediction)

            segment["prediction"] = prediction

    return datasets


def concat_segments_predictions(dataset_exp, dataset_calc):
    exp_segment_num = 0
    segments_exp_predictions = []
    for segment_exp in dataset_exp["segments"]:
        segment_exp["segment_data"]["y"].name = "exp"
        segment_exp["prediction"]["mean"].name = "exp_mean"

        exp_segment_num += 1
        predictions = pd.concat([segment_exp["prediction"]["x"],
                                 segment_exp["segment_data"]["y"],
                                 segment_exp["prediction"]["mean"],
                                 segment_exp["prediction"]["deriv"],
                                 segment_exp["prediction"]["mean_ci"],
                                 segment_exp["prediction"]["mean_se"],
                                 segment_exp["prediction"]["mean_ci_lower"],
                                 segment_exp["prediction"]["mean_ci_upper"],
                                 segment_exp["prediction"]["mean_se_lower"],
                                 segment_exp["prediction"]["mean_se_upper"]],
                                axis=1,)

        predictions.index += 1
        segments_exp_predictions.append(predictions)

    summary_segment_predictions = pd.concat(segments_exp_predictions,
                                            keys = range(1, exp_segment_num+1),
                                            names = ["exp segment number", "index"])

    calc_segment_num = 0
    segments_calc_predictions = []
    for segment_calc in dataset_calc["segments"]:
        if dataset_calc["prediction_points_src"] == "calc":
            calc_values = segment_calc["segment_data"]["y"]
        elif dataset_calc["prediction_points_src"] == "exp":
            calc_values = segment_calc["prediction"]["mean"]

        calc_segment_num += 1
        segments_calc_predictions.append(calc_values)


    summary_segment_calc_predictions = pd.concat(segments_calc_predictions,
                                                 keys = range(1, calc_segment_num+1),
                                                 names = ["calc segment number", "index"])

    try:
        summary_segment_predictions.insert(loc=0,
                                       column="calc segment number",
                                       value=summary_segment_calc_predictions.keys().get_level_values(0).tolist())

        summary_segment_predictions.insert(loc=2,
                                           column="calc",
                                           value=summary_segment_calc_predictions.values)
    except(ValueError):
        raise InputDataError("Amount of exp and calc points do not match")

    summary_segment_predictions.reset_index(level=0, inplace=True)
    return summary_segment_predictions


def save_prediction(out_dir, datasets):
    dataset_exp, dataset_calc = datasets[:]

    summary_predictions = concat_segments_predictions(dataset_exp, dataset_calc)

    segment_exp_name = "{}".format(dataset_exp["name"])
    dproc.save_dataframe_csv(out_dir, segment_exp_name, summary_predictions, sep=";")
