''' CODE FOR TIME-DEPENDENT BOOTSTRAPPED CROSS-VALIDATION
	AND SELECTING MODELS (PARAMETERS AND THRESHOLD)
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

def cat_to_bi_test_helper(df, col_name, train_cols):
	for i in range(len(train_cols)): 
		df[col_name+"_"+str(train_cols[i])] = (df[col_name] == train_cols[i]).astype(int)
	df.drop(col_name, axis=1, inplace=True) 
	return df

def cat_to_bi_test(df, cols, df_train):
	for col in cols:
		df = cat_to_bi_test_helper(df, col, pd.get_dummies(df_train[col]).columns)
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

def fill_str(df, col, value):
	''' filling in arbitrary value '''
	df[col] = df[col].fillna(value)
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
	for label in datelabel:
		df[label] = df[label].astype('datetime64[ns]')
	return df

def get_interval(df, startdate, enddate, labelinterval):
	'''convert into datetime objects before using the function'''
	df[labelinterval] = df[enddate] - df[startdate]
	df[labelinterval] = df[labelinterval].astype('int64')

def get_dummy_dates(df, labelinterval):
	df[labelinterval] = df[labelinterval].apply(lambda x: 1 if x < 0 else 0)

def get_year(df, datelabel):
	for date in datelabel:
		df[date + '_yr'] = df[date].dt.year

def get_month(df, datelabel):
	for date in datelabel:
		df[date + '_mth'] = df[date].dt.month

def applythreshold(array, threshold):
	return np.where(array >= threshold, 1, 0)

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

def sequential_cv_index(df_len, num_moves, total_pct, train_pct):
	'''return train-test split indexes for (num_moves) folds, overlapping allowed'''
	index = 0
	rv = []
	temp = range(df_len)
	train_size = int(np.round((df_len*total_pct)*train_pct))
	test_size = int(np.round(df_len*total_pct)-train_size)
	overlap = int(np.round((num_moves*(train_size+test_size)-df_len)/(num_moves-1)))
	step_size = train_size+test_size-overlap
	if(step_size<test_size):
		print "Warning: test on a single individual for multiple times!"
	for i in range(num_moves):
		if i != num_moves-1:
			train_index = temp[index:index+train_size]
			test_index = temp[index+train_size:index+train_size+test_size]
		else:
			train_index = temp[-(train_size+test_size):-test_size]
			test_index = temp[-test_size:]
		index += step_size
		rv.append((train_index, test_index))
	return rv

def run_cv(train_df, test_df, x_cols, y_col, clf, **kwargs):
	'''train and test the model'''
	from sklearn.preprocessing import StandardScaler
	from time import time
	clf = clf(**kwargs)
	# transforming data and normalization
	X_train = np.array(train_df[x_cols].as_matrix().astype(np.float))
	X_test = np.array(test_df[x_cols].as_matrix().astype(np.float))
	# scaler = StandardScaler()
	# X_train = scaler.fit_transform(X_train)
	# X_test = scaler.fit_transform(X_test)
	y_train = np.ravel(train_df[y_col].astype(np.float))
	y_test = np.ravel(test_df[y_col].astype(np.float))
	# train and test
	begin_train = time()
	model = clf.fit(X_train, y_train)
	end_train = time()
	begin_test = time()
	y_pred = clf.predict(X_test)
	end_test = time()
	y_pred_proba = clf.predict_proba(X_test)
	train_time = end_train - begin_train
	test_time = end_test - begin_test
	return y_pred, y_pred_proba[:,1], y_test, train_time, test_time, model

def evaluate(name, y, y_pred, y_pred_prob, train_time, test_time, threshold):
	'''generate evaluation results'''
	from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, average_precision_score
	rv = {}
	y_pred_new = applythreshold(y_pred_prob, threshold)
	rv["accuracy"] = np.mean(y == y_pred_new)
	rv["precision"] = precision_score(y, y_pred_new)
	rv["recall"] = recall_score(y, y_pred_new)
	rv["f1"] = f1_score(y, y_pred_new)
	rv["auc_roc"] = roc_auc_score(y, y_pred_prob)
	rv["average_precision_score"] = average_precision_score(y,y_pred_prob)
	rv["train_time"] = train_time
	rv["test_time"] = test_time
	return pd.Series(rv), confusion_matrix(y, y_pred_new)

def value_combinations(dic):
	'''
	Takes a dictionary which maps from parameter name to a list of possible
	values, and returns a list of all possible dictionaries. Each dictionary 
	maps from parameter name to one particular value.
	'''
	rv = []
	key = []
	vals = []
	for item in dic.items():
		key.append(item[0])
		vals.append(item[1])
	combs = iter.product(*vals)
	for comb in combs:
		temp = {}
		for i in range(len(key)):
			temp[key[i]] = comb[i]
		rv.append(temp)
	return rv

def select_parameter(train_df, test_df, classifier, x_cols, y_col, dic_param_vals, list_threshold, criterion, **kwargs):
	temp = []
	classifier_name = reduce(lambda x,y: x+y, re.findall('[A-Z][^a-z]*', str(classifier).strip("'>")))
	combs = value_combinations(dic_param_vals)
	metrics = pd.Series(["accuracy","precision","recall","f1","auc_roc","average_precision_score","train_time","test_time"])
	results = pd.DataFrame(columns=metrics)
	for comb in combs:
		y_pred, y_pred_prob, y_test, time_train, time_test = run_cv(train_df, test_df, x_cols, y_col, classifier, **comb)
		for threshold in list_threshold:
			name = classifier_name+"_"+str(comb)+"_"+str(threshold)
			evaluation_result, conf_matrix = evaluate(name, y_test, y_pred, y_pred_prob, time_train, time_test, threshold)
			print name
			print conf_matrix
			results.loc[name] = evaluation_result
	results.sort(columns=criterion,ascending=False)
	return results

def create_match_feature(col_matching, col_to_match, varname, df):
	df = binary_transform(df,[col_matching, col_to_match])
	df[varname] = (df[col_matching] != df[col_to_match])
	df = binary_transform(df, [varname])
	df[varname] = np.where(df[varname]==1,0,1)
	return df

def get_ab_difference(col, sub_col, varname, df):
	df[varname] = np.abs(df[col] - df[sub_col])

def bootstrap(df, percent):
	size = round(len(df)*percent)
	return np.random.choice(np.array(df.index),size)

def bootstrap_split(train_df, last_train_year, last_test_year, percent, num_splits):
	column_name = "client_enrollment_yr"
	cv_train, cv_test = cv_split(train_df,column_name,last_train_year, last_test_year)
	rv = []
	for i in range(num_splits):
		# bootstrap
		train_index = bootstrap(cv_train, percent)
		test_index = bootstrap(cv_test, percent)
		rv.append((train_index, test_index))
	return rv

if __name__ == '__main__':

	TIME = ["client_enrollment", "client_dob", "client_edd", "NURSE_0_FIRST_HOME_VISIT_DATE", "EarliestCourse",
	"EndDate","HireDate"] 

	missing_cols = ["CLIENT_ABUSE_AFRAID_0_PARTNER", "CLIENT_ABUSE_EMOTION_0_PHYSICAL_",
	"CLIENT_ABUSE_FORCED_0_SEX", "CLIENT_ABUSE_HIT_0_SLAP_LAST_TIM", "CLIENT_ABUSE_HIT_0_SLAP_PARTNER",
	"CLIENT_ABUSE_TIMES_0_ABUSE_WEAPO","CLIENT_ABUSE_TIMES_0_BURN_BRUISE",
	"CLIENT_ABUSE_TIMES_0_HEAD_PERM_I","CLIENT_ABUSE_TIMES_0_HURT_LAST_Y",
	"CLIENT_ABUSE_TIMES_0_PUNCH_KICK_", "CLIENT_ABUSE_TIMES_0_SLAP_PUSH_P",
	"CLIENT_WORKING_0_CURRENTLY_WORKI", "English", "INCOME", "PREPGBMI",
	"Spanish", "highest_educ","NURSE_0_YEAR_COMMHEALTH_EXPERIEN", "NURSE_0_YEAR_MATERNAL_EXPERIENCE",
	"NURSE_0_YEAR_NURSING_EXPERIENCE","nurse_English","nurse_hispanic",
	"nurse_Spanish","nurserace_americanindian_alaskanative","nurserace_asian","nurserace_black",
	"nurserace_nativehawaiian_pacificislander","nurserace_white","other_diseases",
	"govt_financial_assistance", "govt_crisis_intervention", "govt_substance_abuse", 
	"medicaid", "govt_healthcare", "govt_healthcare", "govt_educational_programs", "govt_services", "smoking",
	"alcohol",	"marijuana", "hard_drugs", "physical_abuse", "sexual_abuse",  "AgeRatioIntake"]

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
	"nurserace_nativehawaiian_pacificislander","nurserace_white","other_diseases",
	"govt_financial_assistance", "govt_crisis_intervention", "govt_substance_abuse", 
	"medicaid", "govt_healthcare", "govt_healthcare", "govt_educational_programs", "govt_services", "smoking",
	"alcohol",	"marijuana", "hard_drugs", "physical_abuse", "sexual_abuse",  "AgeRatioIntake", "CLIENT_HEALTH_BELIEF_AVG"]

	NUMERICAL = ["PREPGKG", "PREPGBMI", "age_intake_years", 
	"edd_enrollment_interval_weeks", "gest_weeks_intake","NURSE_0_YEAR_COMMHEALTH_EXPERIEN", 
	"NURSE_0_YEAR_MATERNAL_EXPERIENCE", "NURSE_0_YEAR_NURSING_EXPERIENCE","NurseAgeIntake", "AgeRatioIntake"]

	CATEGORICAL = ["MomsRE", "HSGED", "INCOME", "MARITAL", 
	"CLIENT_ABUSE_TIMES_0_HURT_LAST_Y", "CLIENT_ABUSE_TIMES_0_SLAP_PUSH_P",
	"CLIENT_ABUSE_TIMES_0_PUNCH_KICK_", "CLIENT_ABUSE_TIMES_0_BURN_BRUISE",
	"CLIENT_ABUSE_TIMES_0_HEAD_PERM_I", "CLIENT_ABUSE_TIMES_0_ABUSE_WEAPO",
	"CLIENT_BIO_DAD_0_CONTACT_WITH", "CLIENT_LIVING_0_WITH", "CLIENT_WORKING_0_CURRENTLY_WORKI",
	"CLIENT_ABUSE_HIT_0_SLAP_PARTNER","highest_educ", "educ_currently_enrolled_type",
	"SERVICE_USE_0_OTHER1_DESC","SERVICE_USE_0_OTHER2_DESC",
	"SERVICE_USE_0_OTHER3_DESC","SERVICE_USE_0_TANF_CLIENT",
	"SERVICE_USE_0_FOODSTAMP_CLIENT","SERVICE_USE_0_SOCIAL_SECURITY_CL",
	"SERVICE_USE_0_UNEMPLOYMENT_CLIEN", "SERVICE_USE_0_IPV_CLIENT","SERVICE_USE_0_CPS_CHILD",
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
	"PrimRole","SecRole", "CLIENT_HEALTH_BELIEF_AVG"]
	
	CATEGORICAL_2 = ["MomsRE", "HSGED", "INCOME", "MARITAL", 
	"CLIENT_ABUSE_TIMES_0_HURT_LAST_Y", "CLIENT_ABUSE_TIMES_0_SLAP_PUSH_P",
	"CLIENT_ABUSE_TIMES_0_PUNCH_KICK_", "CLIENT_ABUSE_TIMES_0_BURN_BRUISE",
	"CLIENT_ABUSE_TIMES_0_HEAD_PERM_I", "CLIENT_ABUSE_TIMES_0_ABUSE_WEAPO",
	"CLIENT_BIO_DAD_0_CONTACT_WITH", "CLIENT_LIVING_0_WITH", "CLIENT_WORKING_0_CURRENTLY_WORKI",
	"CLIENT_ABUSE_HIT_0_SLAP_PARTNER","highest_educ", "educ_currently_enrolled_type",
	"Highest_Nursing_Degree","Highest_Non_Nursing_Degree","NurseRE",
	"PrimRole","SecRole", "CLIENT_HEALTH_BELIEF_AVG"]

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
	"nurserace_asian", "nurserace_black","nurserace_nativehawaiian_pacificislander","nurserace_white", "govt_financial_assistance", "govt_crisis_intervention", "govt_substance_abuse", 
	"medicaid", "govt_healthcare", "govt_healthcare", "govt_educational_programs", "govt_services", "smoking",	"alcohol",	"marijuana", "hard_drugs", "physical_abuse", "sexual_abuse"]

	# upload data
	input_file = "project_data12.csv"
	df_in = readcsv_funct(input_file)
	# drop rows where premature values are missing
	df = df_in.dropna(subset = ['premature'])
	# transform data forms
	df["edd_enrollment_interval_weeks"]=df["edd_enrollment_interval_weeks"].str.replace(',', '').astype(float)
	df["NURSE_0_BIRTH_YEAR"] = df["NURSE_0_BIRTH_YEAR"].str.replace(',', '').astype(float)

	# filling in missing dates with mode and get years and months
	df = fill_str(df, "client_enrollment", "2009-04-08 00:00:00")
	df = fill_str(df, "client_dob", "1990-08-04 00:00:00")
	df = fill_str(df, "client_edd", "2009-09-15 00:00:00")
	df = fill_str(df, "NURSE_0_FIRST_HOME_VISIT_DATE","2001-06-18 00:00:00")
	df = fill_str(df, "EarliestCourse", "2014-06-13 00:00:00")
	df = fill_str(df, "EndDate", "2010-12-06 00:00:00")
	df = fill_str(df, "HireDate", "2008-01-02 00:00:00")
	df = fill_str(df, "NURSE_0_BIRTH_YEAR", 1963)
	df = fill_str(df, "edd_enrollment_interval_weeks", 21.7) 
	change_time_var(df,TIME)
	get_year(df, TIME)
	get_month(df, TIME)

	# generated features
	get_interval(df, "client_edd", "EndDate", "leftbeforebirth")
	get_interval(df, "client_enrollment", "client_edd", "enrollment_duration")
	get_interval(df, "client_dob_yr", "client_enrollment_yr", "age")
	get_interval(df, "HireDate", "EndDate", "nurse_work_duration")
	get_dummy_dates(df,"leftbeforebirth")
	create_match_feature("English", "nurse_English", "EnglishMatch", df)
	create_match_feature("Spanish", "nurse_Spanish", "SpanishMatch", df)
	get_ab_difference("age_intake_years", "NurseAgeIntake", "MotherNurseAgeDiff", df)
	GENERATED = ["leftbeforebirth", "enrollment_duration", "age", "nurse_work_duration", "EnglishMatch", "SpanishMatch", "MotherNurseAgeDiff",
	"whiteMatch", "hispanicMatch", "blackMatch"]

	# drop the time variables after extracting dates (years and months)
	cols_to_drop = ["client_dob", "client_edd", "client_enrollment", 
	"NURSE_0_FIRST_HOME_VISIT_DATE", "EarliestCourse", "EndDate","HireDate",
	'SERVICE_USE_0_OTHER1_DESC', 'SERVICE_USE_0_OTHER2_DESC',
	'SERVICE_USE_0_OTHER3_DESC', 'SERVICE_USE_0_TANF_CLIENT',
	'SERVICE_USE_0_FOODSTAMP_CLIENT', 'SERVICE_USE_0_SOCIAL_SECURITY_CL',
	'SERVICE_USE_0_UNEMPLOYMENT_CLIEN', 'SERVICE_USE_0_IPV_CLIENT',
	'SERVICE_USE_0_CPS_CHILD', 'SERVICE_USE_0_MENTAL_CLIENT',
	'SERVICE_USE_0_SMOKE_CLIENT', 'SERVICE_USE_0_ALCOHOL_ABUSE_CLIE',
	'SERVICE_USE_0_DRUG_ABUSE_CLIENT', 'SERVICE_USE_0_MEDICAID_CLIENT',
	'SERVICE_USE_0_MEDICAID_CHILD', 'SERVICE_USE_0_SCHIP_CLIENT',
	'SERVICE_USE_0_SCHIP_CHILD', 'SERVICE_USE_0_SPECIAL_NEEDS_CHIL',
	'SERVICE_USE_0_PCP_CLIENT', 'SERVICE_USE_0_PCP_WELL_CHILD',
	'SERVICE_USE_0_DEVELOPMENTAL_DISA', 'SERVICE_USE_0_WIC_CLIENT',
	'SERVICE_USE_0_CHILD_CARE_CLIENT', 'SERVICE_USE_0_JOB_TRAINING_CLIEN',
	'SERVICE_USE_0_HOUSING_CLIENT', 'SERVICE_USE_0_TRANSPORTATION_CLI',
	'SERVICE_USE_0_PREVENT_INJURY_CLI', 'SERVICE_USE_0_BIRTH_EDUC_CLASS_C',
	'SERVICE_USE_0_LACTATION_CLIENT', 'SERVICE_USE_0_GED_CLIENT',
	'SERVICE_USE_0_HIGHER_EDUC_CLIENT', 'SERVICE_USE_0_CHARITY_CLIENT',
	'SERVICE_USE_0_LEGAL_CLIENT', 'SERVICE_USE_0_OTHER1',
	'SERVICE_USE_0_OTHER2', 'SERVICE_USE_0_OTHER3',
	'SERVICE_USE_0_PRIVATE_INSURANCE_', 'SERVICE_USE_0_PRIVATE_INSURANCE1',
	'Nurse_ID', 'NURSE_0_BIRTH_YEAR']

	################################ bootstrap cv ###############################
	# split data into training and test
	last_train_year = 2009 #so means test_df starts from 2010
	column_name = "client_enrollment_yr"
	train_df, test_df = train_test_split(df,column_name,last_train_year)
	
	# create missing indicators
	df_mind_train = run_missing_indicator(train_df,missing_cols)

	# fill_nans
	for col_name in NANCOLS_CAT_BINARY:
		df_mind_train= fill_mode(df_mind_train,col_name)

	for col_name in NUMERICAL:
		if col_name != 'edd_enrollment_interval_weeks':
			df_mind_train = fill_median(df_mind_train,col_name)

	df_mind_train = fill_median(df_mind_train,"MotherNurseAgeDiff")
	
	# Transforming and generating features 
	df_train = cat_var_to_binary(df_mind_train,CATEGORICAL_2)
	df_train = binary_transform(df_train, BOOLEAN)

	create_match_feature("MomsRE_WhiteNH", "nurserace_white", "whiteMatch", df_train)
	create_match_feature("MomsRE_Hispanic or Latina", "nurse_hispanic", "hispanicMatch", df_train)
	create_match_feature("MomsRE_BlackNH", "nurserace_black", "blackMatch", df_train)

	bootstrap_percent = 0.8
	num_splits = 3
	splits = []
	splits += bootstrap_split(df_train, 2007, 2008, bootstrap_percent, num_splits)
	splits += bootstrap_split(df_train, 2008, 2009, bootstrap_percent, num_splits)
	size_list = []
	for i in range(2):
		size_list.append(sum([len(splits[i][0]) for i in range(i*num_splits, (i+1)*num_splits)]))
	total_size = sum(size_list)

	# Models
	# Set dependent variable
	y_col = 'premature'

	# Build classifier and yield predictions
	from sklearn.svm import LinearSVC as LSVC
	from sklearn.ensemble import RandomForestClassifier as RFC
	from sklearn.neighbors import KNeighborsClassifier as KNC
	from sklearn.tree import DecisionTreeClassifier as DTC
	from sklearn.linear_model import LogisticRegression as LR
	from sklearn.ensemble import BaggingClassifier as BC
	from sklearn.ensemble import GradientBoostingClassifier as GBC
	
	# classifiers = [LR, KNC, RFC, DTC, BC, GBC]
	classifiers = [BC]

	import itertools as iter
	dic_param_vals = {
		LR:{"C":[0.01, 0.1, 1.0, 10.0]},
		KNC:{"n_neighbors":[5, 10, 15]},
		LSVC:{"C":[0.1, 1.0, 10.0]},
		RFC:{"n_estimators":[10, 20, 50], "max_features":["auto","log2"], "max_depth":[3, 6, None]},
		DTC:{"criterion":["gini","entropy"],"max_features":["auto","log2",None], "max_depth":[3, 6, None]},
		BC:{"base_estimator":[DTC(), RFC(), GBC()],"n_estimators":[5, 10, 15], "max_samples":[0.5, 0.7, 1.0], "max_features":[0.5, 0.7, 1.0]},
		GBC:{"learning_rate":[0.05, 0.1, 0.3], "n_estimators":[100, 150, 200]}
	}
	
	list_threshold = np.arange(0.25,0.45,0.05)
	metrics = pd.Series(["accuracy","precision","recall","f1","auc_roc","average_precision_score","train_time","test_time"])
	evaluation_results = pd.DataFrame(columns=metrics)
	for classifier in classifiers:
		classifier_name = reduce(lambda x,y: x+y, re.findall('[A-Z][^a-z]*', str(classifier).strip("'>")))
		combs = value_combinations(dic_param_vals[classifier])
		results = pd.DataFrame(columns=metrics)
		for comb in combs:
			fold_index = 1
			sub_results = {}
			for i in list_threshold:
				sub_results[i] = pd.DataFrame(columns=metrics)
			for train_index, test_index in splits:
				cv_train = df_train.loc[train_index]
				cv_test = df_train.loc[test_index]
				for col in cols_to_drop:
					cv_train.drop(col, axis=1, inplace=True)
					cv_test.drop(col, axis=1, inplace=True)
				number_train = (missing(cv_train)>0).sum()
				number_test = (missing(cv_test)>0).sum()
				print "NUMBER OF COLS with missing values in df_train", number_train
				print "NUMBER OF COLS with missing values in df_test", number_test
				x_cols = cv_train.columns[3:]
				y_pred, y_pred_prob, y_test, time_train, time_test, model = run_cv(cv_train, cv_test, x_cols, y_col, classifier, **comb)
				for threshold in list_threshold:
					name = classifier_name+"_"+str(comb)+"_"+str(threshold)+"_"+str(fold_index)
					evaluation_result, conf_matrix = evaluate(name, y_test, y_pred, y_pred_prob, time_train, time_test, threshold)
					print name
					print conf_matrix
					# results.loc[name] = evaluation_result
					sub_results[threshold].loc[name] = evaluation_result
				fold_index += 1
			for i in list_threshold:
				temp = []
				for j in range(2):
					temp.append(sub_results[i][num_splits*j:num_splits*(j+1)].mean()*float(size_list[j])/total_size)
				results.loc[classifier_name+"_"+str(comb)+"_"+str(i)+"_total"] = reduce(lambda x,y: x+y, temp)
				# results.loc[classifier_name+"_"+str(comb)+"_"+str(i)+"_total"] = sub_results[i].mean().copy()
				print results
		results.sort(columns="auc_roc",ascending=False)
		evaluation_results = evaluation_results.append(results)
	baseline = str(1-df_train.describe()[y_col]["mean"])
	baseline_dict = dict(zip(metrics,pd.Series([baseline,0,0,0,0,0,0,0])))
	evaluation_results.loc["baseline"] = baseline_dict
	evaluation_results.to_csv("parametergridsearch.csv")
