
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

input_file = "project_data6.csv"
df_in = readcsv_funct(input_file)
summary_stat= stats(df_in)
df = df_in.dropna(subset = ['premature'])
#print "stats", summary_stat
#Want to convert variables to 0/1 so that can see fraction of types in summary stats


pd.value_counts(df.premature).plot(kind='bar')
col_names = ["premature","MomsRE", "HSGED", "INCOME", "MARITAL","highest_educ", "educ_currently_enrolled_type"]
for col in col_names:
	bar_chart(df,col)

NUMERICAL = ["PREPGKG", "PREPGBMI", "age_intake_years", "edd_enrollment_interval_weeks", "gest_weeks_intake"]

first_graph = df[NUMERICAL]
bin_no = 40
first = dist(first_graph, bin_no, "dist_1.png")
#first[0][0].set_xlim((0,6))
#first[0][1].set_xlim((0, 8))
#plt.savefig("dist_1.png")
plt.show()


'''
bin_no = np.linspace(0,4, 20)
second = dist(second_graph, bin_no, "dist_2.png")
second[0][0].set_xlim((0, 4))
second[0][1].set_xlim((0, 4))
second[1][0].set_xlim((0, 4))
#plt.savefig("dist_2.png")

bin_no = 50
#third_graph[["NumberOfOpenCreditLinesAndLoans"]].plot(kind = 'bar') 
third = dist(third_graph, bin_no, "dist_3.png")
third[0][0].set_xlim((0, 40))
third[0][1].set_xlim((0, 8))
#plt.savefig("dist_3.png")

#monthlyincome = df[["MonthlyIncome"]]
#income_bins = np.linspace(0,50000,30)
#month = dist(monthlyincome, income_bins, "monthly_income.png")
#plt.savefig("Monthly_Income.png")
'''