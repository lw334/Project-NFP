''' MAIN PROJECT PIPELINE TO PUT ALL FUNCTIONS IN FOR NFP PROJECT
'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re

def readcsv_funct(input_csv):
	''' Takes input_csv file name as a string and returns DataFrame
	'''
	df = pd.DataFrame.from_csv(input_csv)
	return df

def missing(dataframe):
	'''Finds number of missing entries for columns in df'''
	missing_data = dataframe.isnull().sum()
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

	# data preprocessing
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
	nul = df[[column_name]].isnull()
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
	new_df = df.copy()
	new_df[column_name] = new_df[column_name].fillna(value)
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
		df[date + '_yr'] = df[date].dt.year

def get_month(df, datelabel):
	for date in datelabel:
		df[date + '_mth'] = df[date].dt.month

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

def sequential_cv(df, total_pct, num_moves, train_pct, x_cols, y_col, clf_class, **kwargs):
	index = 0
	df_len = len(df)
	train_size = np.round((df_len*total_pct)*train_pct)
	test_size = np.round(df_len*total_pct) - train_size
	overlap = np.round((num_moves*(train_size+test_size)-df_len)/(num_moves-1))
	step_size = train_size+test_size-overlap
	for i in range(num_moves):
		if i != num_moves-1:
			train_df = df[index:index+train_size]
			test_df = df[index+train_size:index+train_size+test_size]
		else:
			train_df = df[index:]
			test_df = df[index+train_size:]
		index += step_size
		X_train = np.array(train_df[x_cols].as_matrix())
		X_test = np.array(test_df[x_cols].as_matrix())
		y_train = np.ravel(train_df[y_col])
		y_test = np.ravel(test_df[y_col])
		# train and test
		clf = clf_class(**kwargs)
		begin_train = time()
		clf.fit(X_train, y_train)
		end_train = time()
		begin_test = time()
		y_pred = clf.predict(X_test)
		end_test = time()
		y_pred_prob = clf.predict_proba(X_test)
		train_time = end_train - begin_train
		test_time = end_test - begin_test

def test(df, total_pct, num_moves, train_pct):
	index = 0
	df_len = len(df)
	train_size = np.round((df_len*total_pct)*train_pct)
	test_size = np.round(df_len*total_pct) - train_size
	overlap = np.round((num_moves*(train_size+test_size)-df_len)/(num_moves-1))
	step_size = train_size+test_size-overlap
	for i in range(num_moves):
		if i != num_moves-1:
			train_df = df[index:index+train_size]
			test_df = df[index+train_size:index+train_size+test_size]
		else:
			train_df = df[index:]
			test_df = df[index+train_size:]
		index += step_size
		print train_df, test_df


def run_cv(train_df, test_df, x_cols, y_col, clf_class, **kwargs):
	'''train and test the model'''
	from sklearn.preprocessing import StandardScaler
	from time import time
	# normalization
	X_train = np.array(train_df[x_cols].as_matrix())
	X_test = np.array(test_df[x_cols].as_matrix())
	# scaler = StandardScaler()
	# X_train = scaler.fit_transform(X_train)
	# X_test = scaler.fit_transform(X_test)
	y_train = np.ravel(train_df[y_col])
	y_test = np.ravel(test_df[y_col])
	# train and test
	clf = clf_class(**kwargs)
	begin_train = time()
	clf.fit(X_train, y_train)
	end_train = time()
	begin_test = time()
	y_pred = clf.predict(X_test)
	end_test = time()
	y_pred_proba = clf.predict_proba(X_test)
	train_time = end_train - begin_train
	test_time = end_test - begin_test
	return y_pred, y_pred_proba, y_test, train_time, test_time

def evaluate(name, y, y_pred, y_pred_prob, train_time, test_time):
	#LETS FIX THIS - PUT PRECISION RECALL INTO SEPARATE FUNCTION
	'''generate evaluation results'''
	from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
	rv = {}
	rv["accuracy"] = str(np.mean(y == y_pred))
	rv["precision"] = str(precision_score(y, y_pred))
	rv["recall"] = str(recall_score(y, y_pred))
	rv["f1"] = str(f1_score(y, y_pred))
	rv["auc_roc"] = str(roc_auc_score(y, y_pred))
	#fpr, tpr, _ = roc_curve(y, y_pred_prob)
	# plot_eval_curve(fpr, tpr, name, "roc")
	#rv["auc_roc"] = str(auc(fpr, tpr))
	#precision_c, recall_c, _ = precision_recall_curve(y, y_pred_prob)
	# plot_eval_curve(recall_c, precision_c, name, "prc")
	#rv["auc_prc"] = str(auc(recall_c, precision_c))
	rv["train_time"] = str(train_time)
	rv["test_time"] = str(test_time)
	return pd.Series(rv), confusion_matrix(y, y_pred)

if __name__ == '__main__':

### OUTPUT EVALUATION TABLE
	#upload data
	input_file = "project_data9.csv"
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
	#plt.show()

	# filling in missing dates to "0001-01-01 00:00:00" and get years and months
	TIME = ["client_enrollment", "client_dob", "client_edd", "NURSE_0_FIRST_HOME_VISIT_DATE", "EarliestCourse",
	"HireDate"] 
	#"EndDate" #NEED TO MAKE EndDate ONLY IF BEFORE client_edd
	#NEED TO FIX THIS NURSE BIRTH YEAR "NURSE_0_BIRTH_YEAR"
	MONTH = ["client_edd"]
	if df["client_enrollment"].isnull().any():
		df["client_enrollment"] = df["client_enrollment"].fillna(df["NURSE_0_FIRST_HOME_VISIT_DATE"])
	df = df_in.dropna(subset = TIME)
	#df[TIME] = fill_str(df, TIME, "0001-01-01 00:00:00") 
	change_time_var(df,TIME)
	get_year(df, TIME)
	get_month(df,MONTH)
	df.drop(TIME, axis=1, inplace=True)

	#split data into training and test
	last_train_year = 2009 #so means test_df starts from 2010
	column_name = "client_enrollment_yr"
	train_df, test_df = train_test_split(df,column_name,last_train_year)

	#split train_df into the various train and testing splits for CV
	last_train_year = 2007
	last_test_year = 2008
	column_name = "client_enrollment_yr"
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
	"NURSE_0_YEAR_NURSING_EXPERIENCE","nurse_English","nurse_hispanic",
	"nurse_Spanish","nurserace_americanindian_alaskanative","nurserace_asian","nurserace_black",
	"nurserace_nativehawaiian_pacificislander","nurserace_white","other_diseases"]
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

	BOOLEAN = ["nicu", "premature", "lbw", "CLIENT_ABUSE_EMOTION_0_PHYSICAL_",
	"CLIENT_ABUSE_FORCED_0_SEX", "CLIENT_ABUSE_HIT_0_SLAP_LAST_TIM", "CLIENT_ABUSE_AFRAID_0_PARTNER", "educ_currently_enrolled",
	"live_with_mother", "live_with_FoC_not_spouse", "live_with_spouse_not_FoC",
	"live_with_other_family_members", "live_with_infant_or_child","live_with_other_adults", "income_employment",
	"income_socialsecurity", "income_disability","income_other_public_benefits", "income_other", 
	"English", "Spanish","disease","heart_disease","high_blood_pressure","diabetes",
	"kidney_disease","epilepsy","sickle_cell_disease","chronic_gastrointestinal_disease","asthma_chronic_pulmonary",
	"chronic_urinary_tract_infection","chronic_vaginal_infection_sti","genetic_disease_congenital_anomalies",
	"mental_health","other_diseases","nurse_English","nurse_hispanic",
	"nurse_Spanish","nurserace_americanindian_alaskanative",
	"nurserace_asian", "nurserace_black","nurserace_nativehawaiian_pacificislander","nurserace_white"]
	#NEED DATE COLUMNS OR OTHER RANDOM ONES 
	#fill_nans
	for col_name in NANCOLS_CAT_BINARY:
		df_mind_train= fill_mode(df_mind_train,col_name)
		df_mind_test= fill_mode(df_mind_test,col_name)

	for col_name in NUMERICAL:
		df_mind_train = fill_median(df_mind_train,col_name)
		df_mind_test = fill_median(df_mind_test, col_name)

	# Transforming features 
	df_train = cat_var_to_binary(df_mind_train,CATEGORICAL)
	df_test = cat_var_to_binary(df_mind_test,CATEGORICAL)

	df_train = binary_transform(df_train, BOOLEAN)
	df_test = binary_transform(df_test, BOOLEAN)

	number_train = (missing(df_train)>0).sum()
	number_test = (missing(df_test)>0).sum()
	print "NUMBER OF COLS with missing values in df_train", number_train
	print "NUMBER OF COLS with missing values in df_test", number_test
	

	# df_train = pd.DataFrame.from_csv("train_1.csv")
	# df_test = pd.DataFrame.from_csv("test_1.csv")
	# Models
	# Set dependent and independent variables
	cols_to_drop = ["Nurse_ID", "NURSE_0_BIRTH_YEAR"]
	for col in cols_to_drop:
		df_train.drop(col, axis=1, inplace=True)
		df_test.drop(col, axis=1, inplace=True) 
	y_col = 'premature'
	x_cols = df_train.columns[3:100]

	# Build classifier and yield predictions
	from sklearn.svm import LinearSVC as LSVC
	from sklearn.ensemble import RandomForestClassifier as RFC
	from sklearn.neighbors import KNeighborsClassifier as KNC
	from sklearn.tree import DecisionTreeClassifier as DTC
	from sklearn.linear_model import LogisticRegression as LR
	from sklearn.ensemble import BaggingClassifier as BC
	from sklearn.ensemble import GradientBoostingClassifier as GBC
	# classifiers = [LR, KNC, LSVC, RFC, DTC, BC, GBC]
	classifiers = [RFC]#[LR, RFC, DTC, BC, GBC]
	metrics = pd.Series(["accuracy","precision","recall","f1","auc_roc","train_time","test_time"])#"auc_prc"
	evaluation_result = pd.DataFrame(columns=metrics)
	for classifier in classifiers:
		y_pred, y_pred_prob, y_true, train_time, test_time = run_cv(df_train, df_test, x_cols, y_col, classifier)
		name = reduce(lambda x,y: x+y, re.findall('[A-Z][^a-z]*', str(classifier).strip("'>")))
		dic, conf_matrix = evaluate(name, y_true, y_pred, y_pred_prob, train_time, test_time)
		# print name, conf_matrix
		evaluation_result.loc[name] = dic
	baseline = str(1-df_test.describe()[y_col]["mean"])
	baseline_dict = dict(zip(metrics,pd.Series([baseline,0,0,0,0,0,0])))
	evaluation_result.loc["baseline"] = baseline_dict
	### OUTPUT EVALUATION TABLE
	# print evaluation_result
	# print "Baseline: "+baseline



