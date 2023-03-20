# Imports and Helper Functions
# data Analysis
import pandas as pd
import numpy as np
import random as rng

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#SciKit Learn Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

df_mma1=pd.read_csv('/Users/sagartripathi/Documents/pythonProject/MMA_Betting/data/masterdataframe.csv')
print(df_mma1.head(5))

#creating cloumn named as gender
def get_gender(division):
    if 'Women' in division:
        return 'Female'
    elif 'Heavyweight' in division:
        return 'Male'
    else:
        return 'Male' # assume all other divisions are for male fighters

# create a new gender column based on the division column
df_mma1['gender'] = df_mma1['division'].apply(get_gender)
class_distribution=['Volume_Striker', 'Controller_Wrestler' ,'Knockout_Artist','Submission_Specialist','Hybrid_Finisher']

#dataset description & information
print(df_mma1.describe())
print(df_mma1.info())

columns_to_keep = ['date','gender','result','fighter','opponent','division','stance','dob','method',
                   'total_comp_time','round','time','reach','height','age','knockdowns',
                   'sub_attempts','reversals','control','takedowns_landed','takedowns_attempts',
                   'sig_strikes_landed','sig_strikes_attempts','total_strikes_attempts','avg_total_strikes_attempts',
                   'avg_control','age_differential','avg_knockdowns','sub_attempts_per_min',
                   'sig_strikes_attempts_per_min','takedowns_attempts_per_min',
                   'avg_sub_attempts','avg_reversals','avg_takedowns_attempts',
                   'avg_sig_strikes_landed','avg_sig_strikes_attempts','avg_total_strikes_landed',
                   'avg_total_strikes_attempts']

df_mma1=df_mma1.drop(columns=[col for col in df_mma1.columns if col not in columns_to_keep])
print("---"*50)
df_mma1 = df_mma1.drop(index=range(1000))
print('\n Shape of the dataframe',df_mma1.shape)
print("---"*50)

#data imputations

#.height
height_values = {'Edward Faaloloto': 69, 'Tom Blackledge': 72}

# impute the height values for Edward Faaloloto and Tom Blackledge
for name, height in height_values.items():
    df_mma1.loc[df_mma1['fighter'] == name, 'height'] = height

#2.reach
df_mma1['reach'].fillna(df_mma1['height'], inplace=True)


#3 dob
null_dob_records = df_mma1[df_mma1['dob'].isnull()]
y=null_dob_records[['fighter','opponent']]
records_for_dob=[]
for i,j in y.values:
    records_for_dob.append(df_mma1[(df_mma1['fighter']==j )& (df_mma1['opponent']==i)]['dob'])
df_mma1.loc[df_mma1['dob'].isnull(), 'dob'] = records_for_dob

#4 Age

import pandas as pd
from datetime import datetime

# create sample DataFrame

# convert date columns to datetime format
# df_mma1['date'] = pd.to_datetime(df_mma1['date'])
# df_mma1['date_of_birth'] = pd.to_datetime(df_mma1['dob'])
#
# # calculate age
# df_mma1['age'] = (df_mma1['date'] - df_mma1['dob']).astype('timedelta64[Y]')
#
# # replace null values
# most_common_age = df_mma1['age'].mode()[0]
# df_mma1['age'].fillna(most_common_age, inplace=True)


#5 stance

most_common_value = df_mma1['stance'].mode()[0]
df_mma1['stance'].fillna(most_common_value, inplace=True)

print(df_mma1.isnull().sum().sort_values())

#checking null values after data imputation
print(df_mma1.isnull().sum().sort_values())
print(print("---"*50))

##grouping based on weight class and gender grouping the fighters
average_vol_stk = df_mma1.groupby(['division', 'gender']).agg({'sig_strikes_landed': 'mean'})
# print the resulting DataFrame for average_vol_stk
print(average_vol_stk)

average_cont_wrestler = df_mma1.groupby(['division', 'gender']).agg({'control': 'mean'})
# print the resulting DataFrame for average_cont_wrestler
print(average_cont_wrestler)

average_Submissions = df_mma1.groupby(['division', 'gender']).agg({'sub_attempts': 'mean'})
# print the resulting DataFrame for average_Submission
print(average_Submissions)

average_Knock_down = df_mma1.groupby(['division', 'gender']).agg({'knockdowns': 'mean'})
# print the resulting DataFrame average_Knock_down
print(average_Knock_down)

################## classifying fighter on the basis of the calculated averages  #####################
# A fighter who throws a high volume of strikes, averaging 4 or more significant strikes per minute,
# and has a lower takedown and submission attempt rate.
#
# def fighter_type(avg_strikes, avg_control, avg_submissions):
#     if avg_strikes >= 4 and avg_control < 3 and avg_submissions < 1:
#         return "Volume Striker"
#     elif avg_control >= 3 and avg_submissions < 1 and avg_strikes < 2:
#         return "Controller Wrestler"
#     elif avg_strikes >= 1 and avg_submissions < 1 and avg_control < 2:
#         return "Knockout Artist"
#     elif avg_submissions >= 1 and avg_control < 2 and avg_strikes < 2:
#         return "Submission Specialist"
#     elif 2 <= avg_strikes < 4 and 2 <= avg_control < 3 and avg_submissions >= 1:
#         return "Hybrid Finisher"
#     else:
#         return "Unclassified"
#
# df_mma1['Fighter Type'] = df_mma1.apply(lambda row: fighter_type(row['avg_total_strikes_attempts'], row['avg_control'], row['avg_sub_attempts']), axis=1)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# print(df_mma1)


# check repeated names
name_counts = df_mma1['fighter'].value_counts()
names_to_find = name_counts[name_counts > 1].index.tolist()

# #first we found the fighters names with different divisions they are fighting
matching_records = df_mma1[df_mma1['fighter'].isin(names_to_find)]
# print("*"*50)
# print(matching_records[['fighter','division','gender']].sort_values(by='fighter', ascending=True))
# # second step separate those fighters with multiple divisions
# #find the most common division for the fighters and replace the uncommon with the common division


# Group by fighter name and count unique divisions
fighter_div_counts = matching_records.groupby('fighter')['division'].nunique()

# Filter out fighters who have only fought in one division
multi_div_fighters = fighter_div_counts[fighter_div_counts > 1].index

# For each multi-division fighter, find the most common division
for fighter in multi_div_fighters:
    divisions = matching_records[matching_records['fighter'] == fighter]['division']
    common_division = divisions.value_counts().idxmax()

# Replace uncommon divisions with the common division
    matching_records.loc[(matching_records['fighter'] == fighter) &
                         (matching_records['division'] != common_division),
                         'division'] = common_division

print(matching_records[['fighter', 'division', 'gender']].sort_values(by='fighter', ascending=True))
# df_mma1['division']=df_mma1['fighter'].map(matching_records['division'])


#

df_mma1.to_csv('updated_fighters.csv', index=False)

############ EDA ##################################

#data modelling

 #We Store prediction of each model in our dict
# Helper Functions for our models.

def percep(X_train,Y_train,X_test,Y_test,Models):
    perceptron = Perceptron(max_iter = 1000, tol = 0.001)
    perceptron.fit(X_train, Y_train)
    Y_pred = perceptron.predict(X_test)
    Models['Perceptron'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred)]
    return

def ranfor(X_train,Y_train,X_test,Y_test,Models):
    randomfor = RandomForestClassifier(max_features="sqrt",
                                       n_estimators = 700,
                                       max_depth = None,
                                       n_jobs=-1
                                      )
    randomfor.fit(X_train,Y_train)
    Y_pred = randomfor.predict(X_test)
    Models['Random Forests'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred)]
    return

def dec_tree(X_train,Y_train,X_test,Y_test,Models):
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)
    Y_pred = decision_tree.predict(X_test)
    Models['Decision Tree'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred)]
    return

def SGDClass(X_train,Y_train,X_test,Y_test,Models):
    sgd = SGDClassifier(max_iter = 1000, tol = 0.001)
    sgd.fit(X_train, Y_train)
    Y_pred = sgd.predict(X_test)
    Models['SGD Classifier'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred)]
    return

def linSVC(X_train,Y_train,X_test,Y_test,Models):
    linear_svc = LinearSVC()
    linear_svc.fit(X_train, Y_train)
    Y_pred = linear_svc.predict(X_test)
    Models['SVM'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred)]
    return

def bayes(X_train,Y_train,X_test,Y_test,Models):
    gaussian = GaussianNB()
    gaussian.fit(X_train, Y_train)
    Y_pred = gaussian.predict(X_test)
    Models['Bayes'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred)]
    return

def Nearest(X_train,Y_train,X_test,Y_test,Models):
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    Models['KNN'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred)]

def run_all_and_Plot(df):
    Models = dict()
    from sklearn.model_selection import train_test_split
    X_all = df.drop(['winner'], axis=1)
    y_all = df['winner']
    X_train, X_test, Y_train, Y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=0)
    percep(X_train,Y_train,X_test,Y_test,Models)
    ranfor(X_train,Y_train,X_test,Y_test,Models)
    dec_tree(X_train,Y_train,X_test,Y_test,Models)
    SGDClass(X_train,Y_train,X_test,Y_test,Models)
    linSVC(X_train,Y_train,X_test,Y_test,Models)
    bayes(X_train,Y_train,X_test,Y_test,Models)
    Nearest(X_train,Y_train,X_test,Y_test,Models)
    return Models


def plot_bar(dict):
    labels = tuple(dict.keys())
    y_pos = np.arange(len(labels))
    values = [dict[n][0] for n in dict]
    plt.bar(y_pos, values, align='center', alpha=0.5)
    plt.xticks(y_pos, labels,rotation='vertical')
    plt.ylabel('accuracy')
    plt.title('Accuracy of different models')
    plt.show()


def plot_cm(dict):
    count = 1
    fig = plt.figure(figsize=(10,10))
    for model in dict:
        cm = dict[model][1]
        labels = ['W','L','N','D']
        ax = fig.add_subplot(4,4,count)
        cax = ax.matshow(cm)
        plt.title(model,y=-0.8)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        # plt.subplot(2,2,count)
        count+=1
    plt.tight_layout()
    plt.show()








