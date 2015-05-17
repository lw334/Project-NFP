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

def run_missing_indicator(df, cols):
	for col in cols:
		df = missing_indicator(df,col)
	return df

def fill_nans(df, column_name, value):
	'''fill NaNs with value'''
	new_df = df
	new_df[new_df[column_name].isnull()] = new_df[new_df[column_name].isnull()].fillna(value)
	return new_df 

def fill_str(df, column_name, value):
	''' filling in arbitrary value '''
	df = df[column_name].fillna(value)
	return df

def fill_mode(df, column_name):
	'''fills NaNs with mode of the column for categorical and true/false'''
	new_df = fill_nans(df, column_name,df[column_name].mode())
	return new_df 

def fill_median(df, column_name):
	'''fills NaNs with median of the column for numerical'''
	new_df = fill_nans(df, column_name,df[column_name].median())
	return new_df 

def change_time_var(df, datelabel):
	'''fill in na before conversion'''
	df[datelabel] = df[datelabel].astype('datetime64[ns]')

def get_interval(df, startdate, enddate, labelinterval):
	'''convert into datetime objects before using the function'''
	df[labelinterval] = df[enddate] - df[startdate]
	df[labelinterval] = df[labelinterval].dt.days

def get_year(df, datelabel):
	for date in datelabel:
		df[date + 'yr'] = df[date].dt.year

def get_month(df, datelabel):
	for date in datelabel:
		df[date + 'yr'] = df[date].dt.month

def train_test_split(df,column_name,last_train_yr):
	'''split function for main train and testing, according to last_train_yr'''
	train_df = df.loc[(df[column_name] <= last_train_yr)]
	test_df = df.loc[(df[column_name] > last_train_yr)]
	return train_df, test_df

def cv_split(df,column_name,last_train_yr, last_test_yr):
	'''split function for main train and testing, according to start_time and end_time'''
	train_df = df.loc[(df[column_name] <= last_train_yr)]
	test_df = df.loc[(df[column_name] > last_train_yr) & (df[column_name] <= last_test_yr)]
	return train_df, test_df

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
NUMERICAL = ["PREPGKG", "PREPGBMI", "age_intake_years", "edd_enrollment_interval_weeks", "gest_weeks_intake","NURSE_0_YEAR_COMMHEALTH_EXPERIEN", "NURSE_0_YEAR_MATERNAL_EXPERIENCE",
"NURSE_0_YEAR_NURSING_EXPERIENCE"]
first_graph = df[NUMERICAL]
bin_no = 40
first = dist(first_graph, bin_no, "dist_1.png")
plt.savefig("dist_1.png")
plt.show()

# filling in missing dates to "0001-01-01 00:00:00" and get years and months
TIME = "client_enrollment", "client_dob", "client_edd", "NURSE_0_FIRST_HOME_VISIT_DATE", "EarliestCourse",
"EndDate","HireDate","NURSE_0_BIRTH_YEAR"]
df = fill_str(df, TIME, "0000-00-00 00:00:00")
change_time_var(df,TIME)
get_year(df, TIME)

#split data into training and test
last_train_year = 2009 #so means test_df starts from 2010
train_df, test_df = train_test_split(df,last_train_year)

#split train_df into the various train and testing splits for CV
last_train_year = 2007
last_test_year = 2008
cv_train, cv_test = cv_split(train_df,column_name,last_train_year, last_test_year)

#impute 
#make dummy indicators for columns with large numbers of missing values
missing_cols = ["CLIENT_ABUSE_AFRAID_0_PARTNER", "CLIENT_ABUSE_EMOTION_0_PHYSICAL_",
"CLIENT_ABUSE_FORCED_0_SEX", "CLIENT_ABUSE_HIT_0_SLAP_LAST_TIM", "CLIENT_ABUSE_HIT_0_SLAP_PARTNER",
"CLIENT_ABUSE_TIMES_0_ABUSE_WEAPO","CLIENT_ABUSE_TIMES_0_BURN_BRUISE",
"CLIENT_ABUSE_TIMES_0_HEAD_PERM_I","CLIENT_ABUSE_TIMES_0_HURT_LAST_Y",
"CLIENT_ABUSE_TIMES_0_PUNCH_KICK_", "CLIENT_ABUSE_TIMES_0_SLAP_PUSH_P",
"CLIENT_WORKING_0_CURRENTLY_WORKI", "English", "INCOME", "PREPGBMI",
"Spanish", "highest_educ","NURSE_0_YEAR_COMMHEALTH_EXPERIEN", "NURSE_0_YEAR_MATERNAL_EXPERIENCE",
"NURSE_0_YEAR_NURSING_EXPERIENCE"]
df_mind_train = run_missing_indicator(cv_train,missing_cols)
df_mind_test = run_missing_indicator(cv_test,missing_cols)

NANCOLS_CAT_BINARY = ["MomsRE", "HSGED", "INCOME", "MARITAL", 
"CLIENT_ABUSE_TIMES_0_HURT_LAST_Y", "CLIENT_ABUSE_TIMES_0_SLAP_PUSH_P",
"CLIENT_ABUSE_TIMES_0_PUNCH_KICK_", "CLIENT_ABUSE_TIMES_0_BURN_BRUISE",
"CLIENT_ABUSE_TIMES_0_HEAD_PERM_I", "CLIENT_ABUSE_TIMES_0_ABUSE_WEAPO",
"CLIENT_BIO_DAD_0_CONTACT_WITH", "CLIENT_LIVING_0_WITH", "CLIENT_WORKING_0_CURRENTLY_WORKI",
"CLIENT_ABUSE_HIT_0_SLAP_PARTNER","highest_educ", "educ_currently_enrolled_type",
"Highest_Nursing_Degree","Highest_Non_Nursing_Degree","NurseRE","PrimRole","SecRole","CLIENT_ABUSE_EMOTION_0_PHYSICAL_", "CLIENT_ABUSE_EMOTION_0_PHYSICAL_",
"CLIENT_ABUSE_FORCED_0_SEX", "CLIENT_ABUSE_HIT_0_SLAP_LAST_TIM", 
"CLIENT_ABUSE_AFRAID_0_PARTNER", "educ_currently_enrolled",
"English", "Spanish", "disease","heart_disease","high_blood_pressure","diabetes","kidney_disease",
"epilepsy","sickle_cell_disease","chronic_gastrointestinal_disease",
"asthma_chronic_pulmonary","chronic_urinary_tract_infection",
"chronic_vaginal_infection_sti","genetic_disease_congenital_anomalies",
"mental_health","other_diseases","nurse_English","nurse_hispanic",
"nurse_Spanish","nurserace_americanindian_alaskanative","nurserace_asian","nurserace_black",
"nurserace_nativehawaiian_pacificislander","nurserace_white","other_diseases"]

NUMERICAL = ["PREPGKG", "PREPGBMI", "age_intake_years", 
"edd_enrollment_interval_weeks", "gest_weeks_intake","NURSE_0_YEAR_COMMHEALTH_EXPERIEN", "NURSE_0_YEAR_MATERNAL_EXPERIENCE",
"NURSE_0_YEAR_NURSING_EXPERIENCE"]

CATEGORICAL = ["MomsRE", "HSGED", "INCOME", "MARITAL", 
"CLIENT_ABUSE_TIMES_0_HURT_LAST_Y", "CLIENT_ABUSE_TIMES_0_SLAP_PUSH_P",
"CLIENT_ABUSE_TIMES_0_PUNCH_KICK_", "CLIENT_ABUSE_TIMES_0_BURN_BRUISE",
"CLIENT_ABUSE_TIMES_0_HEAD_PERM_I", "CLIENT_ABUSE_TIMES_0_ABUSE_WEAPO",
"CLIENT_BIO_DAD_0_CONTACT_WITH", "CLIENT_LIVING_0_WITH", "CLIENT_WORKING_0_CURRENTLY_WORKI",
"CLIENT_ABUSE_HIT_0_SLAP_PARTNER","highest_educ", "educ_currently_enrolled_type",
"SERVICE_USE_0_OTHER1_DESC","SERVICE_USE_0_OTHER2_DESC",
"SERVICE_USE_0_OTHER3_DESC","SERVICE_USE_0_TANF_CLIENT",
"SERVICE_USE_0_FOODSTAMP_CLIENT","SERVICE_USE_0_SOCIAL_SECURITY_CL",
"SERVICE_USE_0_UNEMPLOYMENT_CLIEN",
"SERVICE_USE_0_IPV_CLIENT","SERVICE_USE_0_CPS_CHILD",
"SERVICE_USE_0_MENTAL_CLIENT","SERVICE_USE_0_SMOKE_CLIENT",
"SERVICE_USE_0_ALCOHOL_ABUSE_CLIE","SERVICE_USE_0_DRUG_ABUSE_CLIENT",
"SERVICE_USE_0_MEDICAID_CLIENT","SERVICE_USE_0_MEDICAID_CHILD",
"SERVICE_USE_0_SCHIP_CLIENT","SERVICE_USE_0_SCHIP_CHILD",
"SERVICE_USE_0_SPECIAL_NEEDS_CHIL","SERVICE_USE_0_PCP_CLIENT","SERVICE_USE_0_PCP_WELL_CHILD",
"SERVICE_USE_0_DEVELOPMENTAL_DISA","SERVICE_USE_0_WIC_CLIENT","SERVICE_USE_0_CHILD_CARE_CLIENT",
"SERVICE_USE_0_JOB_TRAINING_CLIEN","SERVICE_USE_0_HOUSING_CLIENT",
"SERVICE_USE_0_TRANSPORTATION_CLI","SERVICE_USE_0_PREVENT_INJURY_CLI",
"SERVICE_USE_0_BIRTH_EDUC_CLASS_C","SERVICE_USE_0_LACTATION_CLIENT",
"SERVICE_USE_0_GED_CLIENT","SERVICE_USE_0_HIGHER_EDUC_CLIENT",
"SERVICE_USE_0_CHARITY_CLIENT","SERVICE_USE_0_LEGAL_CLIENT","SERVICE_USE_0_OTHER1",
"SERVICE_USE_0_OTHER2","SERVICE_USE_0_OTHER3",
"SERVICE_USE_0_PRIVATE_INSURANCE_","SERVICE_USE_0_PRIVATE_INSURANCE1",
"Highest_Nursing_Degree","Highest_Non_Nursing_Degree","NurseRE",
"PrimRole","SecRole"]

#NEED DATE COLUMNS OR OTHER RANDOM ONES 
#fill_nans(

for col_name in NANCOLS_CAT_BINARY:
	df_mind_train= fill_mode(df_mind_train,col_name)
	df_mind_test= fill_mode(df_mind_test,col_name)

for col_name in NUMERICAL:
	df_mind_train = fill_median(df_mind_train,col_name)
	df_mind_test = fill_median(df_mind_test, col_name)

df_train = cat_var_to_binary(df_mind_train,CATEGORICAL)
df_test = cat_var_to_binary(df_mind_test,CATEGORICAL)
#transform features


### ADD MODELS HERE

### OUTPUT EVALUATION TABLE



