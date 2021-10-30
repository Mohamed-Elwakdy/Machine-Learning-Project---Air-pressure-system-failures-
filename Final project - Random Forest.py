
# Import packages
# ---------------

import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.ensemble import forest

from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix


########################### IMPORTANT NOTE ##############################


### Here, I will use the RANDOM FOREST CLASSIFIER without removing all rows in
### training and testing dataset which contain "missing values"

### The highest classification accuracy I get is "99.27 %".  


#########################################################################


# Read the datasets
# -----------------

# Read the training dataset

Training_Dataset = pd.read_csv("aps_failure_training_set.csv")

# Read the testing dataset

Testing_Dataset = pd.read_csv("aps_failure_test_set.csv")

# Convert the dependent variable into 1/0 instead of pos/neg in training dataset

Training_Dataset['target_col'] = np.where(Training_Dataset['class']=="neg", 0,1)

# Convert the dependent variable into 1/0 instead of pos/neg in testing dataset

Testing_Dataset['target_col'] = np.where(Testing_Dataset['class']=="neg", 0,1)

##########

# The percentage the "ones" to "Zeros" of training dataset

Sum = Training_Dataset['target_col'].sum() # The sum of "ones" is 1000
Length = len(Training_Dataset) # Number of rows "60000"

print('proportion:',Sum/Length)


# The percentage the "ones" to "Zeros" of testing dataset

Sum1 = Testing_Dataset['target_col'].sum() # The sum of "ones" is 375
Length1 = len(Testing_Dataset) # Number of rows "16000"

print('proportion:',Sum1/Length1)


############

feat_cols =  ["aa_000",	"ab_000", "ac_000", "ad_000", "ae_000",	"af_000", "ag_000", "ag_001",
             "ag_002",	"ag_003", "ag_004", "ag_005", "ag_006",	"ag_007", "ag_008", "ag_009",
             "ah_000",	"ai_000", "aj_000", "ak_000", "al_000",	"am_0",	"an_000", "ao_000",
             "ap_000",	"aq_000", "ar_000", "as_000", "at_000",	"au_000", "av_000", "ax_000",
             "ay_000",	"ay_001", "ay_002", "ay_003", "ay_004", "ay_005", "ay_006", "ay_007",
             "ay_008",	"ay_009", "az_000", "az_001", "az_002", "az_003", "az_004", "az_005",
             "az_006",	"az_007", "az_008", "az_009", "ba_000",	"ba_001", "ba_002", "ba_003",
             "ba_004",	"ba_005", "ba_006", "ba_007", "ba_008",	"ba_009", "bb_000", "bc_000",
             "bd_000",	"be_000", "bf_000", "bg_000", "bh_000",	"bi_000", "bj_000", "bk_000",
             "bl_000",	"bm_000", "bn_000", "bo_000", "bp_000",	"bq_000", "br_000", "bs_000",
             "bt_000",	"bu_000", "bv_000", "bx_000", "by_000",	"bz_000", "ca_000", "cb_000",
             "cc_000",	"cd_000", "ce_000", "cf_000", "cg_000",	"ch_000", "ci_000", "cj_000",
             "ck_000",	"cl_000", "cm_000", "cn_000", "cn_001",	"cn_002", "cn_003", "cn_004",
             "cn_005",	"cn_006", "cn_007", "cn_008", "cn_009",	"co_000", "cp_000", "cq_000",
             "cr_000",	"cs_000", "cs_001", "cs_002", "cs_003",	"cs_004", "cs_005", "cs_006",
             "cs_007",	"cs_008", "cs_009", "ct_000", "cu_000",	"cv_000", "cx_000", "cy_000",
             "cz_000",	"da_000", "db_000", "dc_000", "dd_000",	"de_000", "df_000", "dg_000",
             "dh_000",	"di_000", "dj_000", "dk_000", "dl_000",	"dm_000", "dn_000", "do_000",
             "dp_000",	"dq_000", "dr_000", "ds_000", "dt_000",	"du_000", "dv_000", "dx_000",
             "dy_000",	"dz_000", "ea_000", "eb_000", "ec_00",	"ed_000", "ee_000", "ee_001",
             "ee_002",	"ee_003", "ee_004", "ee_005", "ee_006",	"ee_007", "ee_008", "ee_009",
             "ef_000",	"eg_000"]


#######

# Delete the variable name "class" in training dataset 

del Training_Dataset['class']

# Delete the variable name "class" in testing dataset 

del Testing_Dataset['class']

#######

#######

# Change the name of the dependent variable in training dataset

Training_Dataset  = Training_Dataset.rename(columns = {"target_col":"class"})

# Change the name of the dependent variable in testing dataset

Testing_Dataset  = Testing_Dataset.rename(columns = {"target_col":"class"})


#######

# Convert "na" to "NaN"
# --------------------

# Because the missing values are not NaN, the "na" will be converted
# to NaN in training and testing dataset

for feat in feat_cols:
    Training_Dataset[feat].replace('na', np.nan, inplace= True)
    Testing_Dataset[feat].replace('na', np.nan, inplace= True)


##################

# Split the training and testing datasets into training and testing data and
# training and testing labels. 

X_Training=Training_Dataset.loc[:,'aa_000':'eg_000']
y_Training =Training_Dataset.loc[:,'class']

X_Testing=Testing_Dataset.loc[:,'aa_000':'eg_000']
y_Testing =Testing_Dataset.loc[:,'class']


# Create our imputer to replace missing values with the mean e.g.

# The SimpleImputer class provides basic strategies for imputing missing values.
# Missing values can be imputed with a provided constant value, or using the
# statistics (mean, median or most frequent) of each column in which the missing
# values are located. This class also allows for different missing values encodings.

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(X_Training )
imp = imp.fit(X_Testing)


# Impute our data, then train
X_train_imp= imp.transform(X_Training)
X_test_imp = imp.transform(X_Testing)


#################################

# Using Random Forest Classifier
# ------------------------------

# "n_jobs = -1" tells the random forest classifier to create a separate job
# it's a seperate process basically for each CPU I have so that's pretty much
# what I want all the time.

# I got an accuracy 0.9926875

m = RandomForestClassifier (n_jobs=-1,random_state=0)

m.fit(X_train_imp,y_Training)

# Make Evalustion of the model using the testing dataset

m.score1 = m.score(X_test_imp, y_Testing)

print ("score1 =",m.score1 )

############

# Bagging
# -------

# Check if I will get a better accuracy by increasing the number of estimators
# Bagging and Random forest are the most commonly used powerful basic ensemble
# techniques.

# If I increase the number of estimators to 10 and 20. I didn't get a better
# classification accuracy.  


m = RandomForestClassifier (n_estimators = 10,n_jobs=-1,random_state=0)

m.fit(X_train_imp,y_Training)

# score = 0.9910625

m.score2 = m.score(X_test_imp, y_Testing)

print ("score2 =",m.score2)


# If I increase the number of estimators to twenty. I didn't get a better
# classification accuracy.

# increasing the number of trees "n_estimators" from 10 to 20 doesn't make that
# much effect on the result

m = RandomForestClassifier (n_estimators = 20,n_jobs=-1,random_state=0)

m.fit(X_train_imp,y_Training)

# m.score is 0.9915625

m.score3 = m.score(X_test_imp, y_Testing)

print ("score3 =",m.score3)

# increasing the number of trees "n_estimators" from 40, I don't get a better
# classification accuracy

m = RandomForestClassifier (n_estimators = 40,n_jobs=-1,random_state=0)

m.fit(X_train_imp,y_Training)

# m.score is 0.9923125

m.score4 = m.score(X_test_imp, y_Testing)

print ("score4 =",m.score4)

################

# SUBSAMPLING
#-------------

# By doing subsamples, the processing will be faster, so let's try to do subsampling and
# see if I will get better results. 


# Unfortunately, Scikit-learn does not support this functionality out of
# the box so I have to write "set_rf_samples".

def set_rf_samples (n):
     forest._generate_sample_indices = (lambda rs, n_samples:
                                       forest.check_random_state(rs).randint(0,n_samples,n))

set_rf_samples (10000)

####

# The out-of-bag (OOB) error is the average error for each calculated using
# predictions from the trees.

# After doing "resampling", I didn't get a better accuracy 

m1 = RandomForestClassifier (n_jobs=-1, oob_score=True,random_state=0)

m1.fit(X_train_imp,y_Training)

# m.score is 0.9926875

m1.score5 = m1.score(X_test_imp, y_Testing)

print ("score5 =",m1.score5)


####

# After doing "resampling", I didn't get a better accuracy and increased the number
# of trees to 40, I didn't get a better accuracy so resumpling doesn't make an
# effect in getting a better model performance. 

m = RandomForestClassifier (n_estimators = 40, n_jobs=-1, oob_score=True,random_state=0)

m.fit(X_train_imp,y_Training)

# m.score is 0.9923125

m.score6 = m.score(X_test_imp, y_Testing)

print ("score6 =",m.score6)

######

# I revert a full bootstrap sample. 
# Calling "reset_rf_samples" to get back to our full data set

def reset_rf_samples():
    forest._generate_sample_indices = (lambda rs, n_samples:
                                       forest.check_random_state(rs).randint(0, n_samples, n_samples))

reset_rf_samples()

######


# The next parameter I can try fidding with is "min_samples_leaf" and
# so "min_sample_leaf" says stop training the tree further when your leaf
# node has 3 or less.

# By using this parameter "min_sample_leaf", I didn't get a better classification
# accuracy 

m = RandomForestClassifier (n_estimators = 20, min_samples_leaf = 3, n_jobs=-1, oob_score=True,random_state=0)

m.fit(X_train_imp,y_Training)

# m.score is 0.9916875

m.score7 = m.score(X_test_imp, y_Testing)

print ("score7 =",m.score7)


#####

# I can also increase the amount of variation amongst the trees by not
# only using a sample of rows for each tree, but also using a sample of
# columns. We do this by specifying "max_features, which is the proportion
# of features to randomly at each split.

m = RandomForestClassifier (n_estimators = 20, min_samples_leaf = 3, n_jobs=-1, max_features = 0.2, oob_score=True,random_state=0)

m.fit(X_train_imp,y_Training)

# m.score is 0.9920625

m.score8 = m.score(X_test_imp, y_Testing)

print ("score8 =",m.score8)

######

# Calculate the cost saving

confmat = confusion_matrix(y_true=y_Testing, y_pred= m1.predict(X_test_imp))
print(confmat)

true_positive_rate = confmat[0][0]  
true_negative_rate = confmat[1][1]
false_positive_rate =confmat[0][1]
false_negative_rate =confmat[1][0]


print("true_positive_rate:",true_positive_rate,"true_negative_rate:",
      true_negative_rate,'(total:',true_negative_rate+true_positive_rate,')')


def cost (false_positive_rate, false_negative_rate):

    cost = 10*false_positive_rate+500*false_negative_rate
    return cost

# Running the cost function to determine how much baseline model cost

cost1 = cost (false_positive_rate, false_negative_rate)

print ("cost1", cost1)


