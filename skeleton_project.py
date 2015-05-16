''' MAIN PROJECT PIPELINE TO PUT ALL FUNCTIONS IN FOR NFP PROJECT
'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def readcsv_funct(input_csv):
	''' Takes input_csv file name as a string and returns DataFrame
	'''
	df = pd.DataFrame.from_csv(input_csv)
	return df

def missing(dataframe):
	'''Finds number of missing entries for columns in df'''
	count_nan = len(df) - df.count()
	missing_data = count_nan
	return missing_data

def stats(dataframe):
	''' Generates data summaries and distributions from DataFrame input'''
	df = dataframe.copy()
	summary_stats = df.describe()
	mode = df.mode()
	summary_stats = summary_stats.append(mode)
	count_nan = pd.Series(len(df) - df.count(), name = "missing_vals").T
	summary_stats = summary_stats.append(count_nan).T
	new_stats = summary_stats.rename(columns = {'50%':'median', 0:'mode'})
	summary_stats = new_stats
	#summary_stats=np.round(new_stats, decimals=2)
	summary_stats.to_csv("summary_stats.csv") 
	return  summary_stats

def dist(dataframe, number_of_bins, name):
	df = dataframe
	ax_list = df.hist(bins=number_of_bins)
	plt.savefig(name)
	return ax_list

def bar_chart(dataframe,col_title):
	df = dataframe
	dataframe_cols = df[col_title]
	bar = pd.value_counts(dataframe_cols).plot(kind='bar',title=col_title)
	name = col_title + ".png"
	plt.savefig(name)
	plt.rcParams.update({'font.size': 12})
	return bar

	# preprocessing of data
def binary_transform(df, cols):
''' Transform True/False to 1/0'''
	df[cols] = df[cols].applymap(lambda x: 1 if x else 0)
	return df

def cat_var_to_binary_helper(df, column_name):
''' function that can take a categorical variable and create binary variables from it. '''
	dummies = pd.get_dummies(df[column_name], prefix=column_name)
	df = df.join(dummies.ix[:,:])
	df.drop(column_name, axis=1, inplace=True) 
	return df

def cat_var_to_binary(df, cols):
''' apply cat_var_to_binary_helper() to a list of columns'''
	for col in cols:
		df = cat_var_to_binary_helper(df, col)
	return df

def missing_indicator(df, column_name):
""" add a missing indicator for a feature to the dataframe, 1 if missing and 0 otherwise. """
	nul = df[[column_name]].  isnull()
	nul = nul.applymap(lambda x: 1 if x else 0)
	name = column_name + "_missing"
	df[name] = nul
	return df

def fill_nans(df, column_name, value):
	'''fill NaNs with value'''
	new_df = df
	new_df[new_df[column_name].isnull()] = new_df[new_df[column_name].isnull()].fillna(value)
	return new_df 

def fill_mode(df, column_name):
	'''fills NaNs with mode of the column for categorical and true/false'''
	new_df = fill_nans(df, column_name,df[column_name].mode())
	return new_df 

def fill_median(df, column_name):
	'''fills NaNs with median of the column for numerical'''
	new_df = fill_nans(df, column_name,df[column_name].median())
	return new_df 



if __name__ == '__main__':

#upload data
input_file = "project_data6.csv"
df_in = readcsv_funct(input_file)
#drop rows where premature values are missing
df = df_in.dropna(subset = ['premature'])
summary_stat= stats(df)
#print "stats", summary_stat

#saves distributions
pd.value_counts(df.premature).plot(kind='bar')
col_names = ["premature","MomsRE", "HSGED", "INCOME", "MARITAL","highest_educ", "educ_currently_enrolled_type"]
for col in col_names:
	bar_chart(df,col)
NUMERICAL = ["PREPGKG", "PREPGBMI", "age_intake_years", "edd_enrollment_interval_weeks", "gest_weeks_intake"]
first_graph = df[NUMERICAL]
bin_no = 40
first = dist(first_graph, bin_no, "dist_1.png")
plt.savefig("dist_1.png")
plt.show()

#split data into training and test


