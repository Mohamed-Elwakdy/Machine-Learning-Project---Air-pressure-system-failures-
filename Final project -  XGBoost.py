
# Import packages
# ---------------

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np

#######################################################

## The highest classification accuracy we get is 99.41% 

#######################################################


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

########

# Check the type of data in training dataset and testing dataset
# ---------------------------------------------------------------

print(Training_Dataset.dtypes)
print(Testing_Dataset.dtypes)


#########


# Convert the data in training and testing datasets to "float"
Training_Dataset = Training_Dataset.astype(float)
Testing_Dataset = Testing_Dataset.astype(float)

#########

# check correlation for training and testing datasets
# ---------------------------------------------------

length_mat = len(feat_cols)
corr_ = Training_Dataset[feat_cols].corr()
print (corr_)

corr_thres = 0.8

for row in list(range(length_mat)):
    for col in list(range(row)):
        corr_val = corr_.iloc[row,col]
        if corr_val > corr_thres:
            print(corr_.index[row],'is correlated with:',corr_.index[col],'with correlation value of',corr_val)


corr1_ = Testing_Dataset[feat_cols].corr()
print (corr1_)

for row in list(range(length_mat)):
    for col in list(range(row)):
        corr_val = corr1_.iloc[row,col]
        if corr_val > corr_thres:
            print(corr1_.index[row],'is correlated with:',corr1_.index[col],'with correlation value of',corr_val)

#########

### Finding so far:
            
# 1- "ao_000", "an_000", "bb_000", "ah_000" seems have high predictive power
# but highly correlated

# 2- a lots for the features are heavily correlated

#########

# Split the training and testing datasets into training and testing data and
# training and testing labels. 

X_Training=Training_Dataset.loc[:,'aa_000':'eg_000']
y_Training =Training_Dataset.loc[:,'class']

X_Testing=Testing_Dataset.loc[:,'aa_000':'eg_000']
y_Testing =Testing_Dataset.loc[:,'class']

##########

# Training the model
# ------------------

model = XGBClassifier()

model.fit(X_Training, y_Training)

##########

# Finding the importantance variables
# -----------------------------------

# Get feature importance in xgboost and sort it with descending.

sorted_idx = np.argsort(model.feature_importances_)[::-1]

# print all sorted importances and the name of columns together as lists

for index in sorted_idx:
    print([X_Training.columns[index], model.feature_importances_[index]])

# Plot the importances with XGboost built-in function
# max_num_features (int, default None) is the Maximum number of top features
# displayed on plot. If None, all features will be displayed.

plot_importance(model, max_num_features = 15) # Plot importance based on fitted
                                              # trees.
plt.show()
    
###########

###########

# Make Evaluation of the model
# ----------------------------

y_pred = model.predict(X_Testing)

predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_Testing, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0)) # Getting accuracy 99.41%

# using the probabilities and getting only one output for each observation.

# I will also look for the GINI metric. In this example, I will learn how 
# GINI model metric is calculated using True Positive Results (TPR) and
# False Positive Results (FPR) values from a given test dataset.

# train gini: 0.9999996101694917
y_train_pred = model.predict_proba(X_Training)[:,1]
print('train gini:',2*roc_auc_score(y_Training,y_train_pred)-1) # GINI metric is

# test gini: 0.9898705919999999                                                                # calculated from AUC
y_test_pred = model.predict_proba(X_Testing)[:,1]
print('test gini:', 2* roc_auc_score(y_Testing,y_test_pred)-1) # GINI metric is
                                                            # calculated from AUC

confmat = confusion_matrix(y_true=y_Testing, y_pred=model.predict(X_Testing))
print(confmat)

true_positive_rate = confmat[0][0]  
true_negative_rate = confmat[1][1]
false_positive_rate =confmat[0][1]
false_negative_rate =confmat[1][0]

# true_positive_rate: 15612 true_negative_rate: 293 (total: 15905 )

print("true_positive_rate:",true_positive_rate,"true_negative_rate:",
      true_negative_rate,'(total:',true_negative_rate+true_positive_rate,')')


############

#for completeness, we can always change the cut-off threshold and get the
#threshold value where we maximize the business objective for example, here
#we will choose the cut-off value where the f1 score is the maximum.   

_ = pd.DataFrame({'flag':y_Testing,'pred':y_test_pred})

cutoff_list = np.linspace(0, 1,endpoint=False, num=100)
f1_list = []

for cutoff in cutoff_list:
    thres = cutoff
    tp = np.where((_['pred']>=thres)&(_['flag']==1),1,0).sum()
    tn = np.where((_['pred']<=thres)&(_['flag']==0),1,0).sum()

    fp = np.where((_['pred']>=thres)&(_['flag']==0),1,0).sum()
    fn = np.where((_['pred']<=thres)&(_['flag']==1),1,0).sum()
    
    f1 = 2*tp/(2*tp + fp + fn)
    f1_list.append(f1)


_ = pd.DataFrame({'cutoff':cutoff_list,'f1':f1_list})
_['f1'].plot()

plt.xlabel('Cutoff', fontsize=13)
plt.ylabel('F1 Score', fontsize=13)
plt.title('Cutoff Versus F1 Score',fontsize=14)
plt.show()

max_tp_tn = _['f1'].max()
optimum_cutoff = _[_['f1'] ==max_tp_tn]['cutoff'].values

# maximum f1: 0.8726287262872628 with cut off proba at: [0.14]
print('maximum f1:',max_tp_tn,'with cut off proba at:',optimum_cutoff)


###############

# Calculate the cost saving

def cost (false_positive_rate, false_negative_rate):

    cost = 10*false_positive_rate+500*false_negative_rate
    return cost

# Running the cost function to determine how much baseline model cost

cost = cost (false_positive_rate, false_negative_rate)

# cost is 41130

print ("cost", cost)


################

# Tuning the hyperparameters is going to be important to minimizing the cost

#n_jobsint, default=None
#Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend
#context. -1 means using all processors. See Glossary for more details.

final_model = XGBClassifier(learning_rate= 0.05,max_depth= 8,n_estimators= 800,subsample= 0.7)
final_model.fit(X_Training, y_Training)

# Make evaluation of the model

y_pred1 = model.predict(X_Testing)

predictions1 = [round(value) for value in y_pred1]

accuracy1 = accuracy_score(y_Testing, predictions1)

print("Accuracy: %.2f%%" % (accuracy1 * 100.0)) # Getting accuracy 99.41%

#########

y_train_pred1 = final_model.predict_proba(X_Training)[:,1]
print('train gini:',2*roc_auc_score(y_Training,y_train_pred1)-1)

y_test_pred1 = final_model.predict_proba(X_Testing)[:,1]
print('test gini:',2*roc_auc_score(y_Testing,y_test_pred1)-1)

confmat1 = confusion_matrix(y_true=y_Testing, y_pred=final_model.predict(X_Testing))
print(confmat1)
true_positive_rate1 = confmat1[0][0]  
true_negative_rate1 = confmat1[1][1]
false_positive_rate1 =confmat1[0][1]
false_negative_rate1 =confmat1[1][0]  

print("true_positive_rate:",true_positive_rate1,"true_negative_rate:",
      true_negative_rate1,'(total:',true_negative_rate1+true_positive_rate1,')')

############

# Calculat the cost saving

def cost1 (false_positive_rate1, false_negative_rate1):

    cost1 = 10*false_positive_rate1+500*false_negative_rate1
    return cost1

# Running the cost function to determine how much baseline model cost

cost1 = cost1 (false_positive_rate1, false_negative_rate1)

# We can reduce the cost when "max_depth = 8" and number of estimators is "800"
# The cost is 39660

print ("cost1", cost1)



