import os

def save_dataframe_csv(dir_name, dataframe_name, dataframe, header=True, sep=","):
	file_path = os.path.join(dir_name, dataframe_name + ".csv")
	dataframe.to_csv(file_path, index=False, header=header, sep=sep)
