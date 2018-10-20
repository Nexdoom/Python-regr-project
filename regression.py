import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import sympy as sp
import sympy.parsing.sympy_parser as sp_parser
import os

N_PREDICT_POINTS = 100
FLOAT_REGEX = r"[+-]?([0-9]*[.])?[0-9]+"

def load_data_csv(data_dir, data_name, sep=","):
	data_file_name = data_name + ".csv"
	data_file_path = os.path.join(data_dir, data_file_name)
	data = pd.read_csv(data_file_path, names=["x","y"], sep=sep)
	data = data.sort_values("x").reset_index(drop=True)
	return data

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

	#print("\n==============================================================================")
	#print('=== Coefs ===\n', regr_result.params)
	#print("\n=== Coefs std errors ===\n", regr_result.bse)
	#print("\n=== Coefs confidence interval (P=0.95) ===:\n", regr_result.conf_int(alpha=0.05))
	#print("==============================================================================")

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
