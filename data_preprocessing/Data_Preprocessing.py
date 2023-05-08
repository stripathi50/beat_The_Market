# Imports and Helper Functions
# data Analysis
import pandas as pd
import numpy as np
import random as rng
from datetime import datetime
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt



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
                   'sub_attempts','reversals','control','takedowns_attempts',
                   'total_strikes_attempts','avg_control','avg_knockdowns',
                   'sub_attempts_per_min','sig_strikes_attempts_per_min','takedowns_attempts_per_min',
                   'avg_sub_attempts','avg_reversals','avg_takedowns_attempts',
                   'avg_sig_strikes_landed','avg_sig_strikes_attempts',
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
df_mma1['date'] = pd.to_datetime(df_mma1['date'])
df_mma1['dob']=pd.to_datetime(df_mma1['dob'])
df_mma1['year'] = df_mma1['date'].dt.year
df_mma1['Dob_year'] = df_mma1['date'].dt.year

def calculate_age(row):
    if pd.isnull(row['age']):
        return row['year'] - row['Dob_year']
    else:
        return row['age']

# apply the function to the DataFrame to replace null values in the "age" column
df_mma1['age'] = df_mma1.apply(calculate_age, axis=1)
df_mma1=df_mma1.drop(['Dob_year'], axis=1)

#5 stance
most_common_value = df_mma1['stance'].mode()[0]
df_mma1['stance'].fillna(most_common_value, inplace=True)
print(df_mma1.isnull().sum().sort_values())

#checking null values after data imputation
print(df_mma1.isnull().sum().sort_values())
print(print("---"*50))

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# check repeated names
name_counts = df_mma1['fighter'].value_counts()
names_to_find = name_counts[name_counts > 1].index.tolist()

#first we found the fighters names with different divisions they are fighting
matching_records = df_mma1[df_mma1['fighter'].isin(names_to_find)]

# Group by fighter name and count unique divisions
fighter_div_counts = matching_records.groupby('fighter')['division'].nunique()

# Filter out fighters who have only fought in one division
multi_div_fighters = fighter_div_counts[fighter_div_counts > 1].index

# For each multi-division fighter, find the most common division
for fighter in multi_div_fighters:
    divisions = matching_records[matching_records['fighter'] == fighter]['division']
    common_division = divisions.value_counts().idxmax()

# Replace uncommon divisions with the common division
    df_mma1.loc[(df_mma1['fighter'] == fighter) &
                         (df_mma1['division'] != common_division),
                         'division'] = common_division



################## classifying fighter on the basis of the calculated averages  #####################

#per fighter per divsion
average_vol_stk_per_fighter = df_mma1.groupby(['fighter','division']).agg({'total_strikes_attempts': 'mean'})
# print the resulting DataFrame for average_vol_stk
print(average_vol_stk_per_fighter)

average_cont_wrestler_per_fighter = df_mma1.groupby(['fighter','division']).agg({'control': 'mean'})
# print the resulting DataFrame for average_cont_wrestler
print(average_cont_wrestler_per_fighter)

average_Submissions_per_fighter = df_mma1.groupby(['fighter','division']).agg({'sub_attempts': 'mean'})
# print the resulting DataFrame for average_Submission
print(average_Submissions_per_fighter)

average_Knock_down_per_fighter = df_mma1.groupby(['fighter','division']).agg({'knockdowns': 'mean'})
# print the resulting DataFrame average_Knock_down
print(average_Knock_down_per_fighter)

# per division
average_vol_stk_per_div = df_mma1.groupby(['division']).agg({'total_strikes_attempts': 'mean'})
# print the resulting DataFrame for average_vol_stk
print(average_vol_stk_per_div)

average_cont_wrestler_per_div = df_mma1.groupby(['division']).agg({'control': 'mean'})
# print the resulting DataFrame for average_cont_wrestler
print(average_cont_wrestler_per_div)

average_Submissions_per_div = df_mma1.groupby(['division']).agg({'sub_attempts': 'mean'})
# print the resulting DataFrame for average_Submission
print(average_Submissions_per_div)

average_Knock_down_per_div = df_mma1.groupby(['division']).agg({'knockdowns': 'mean'})
# print the resulting DataFrame average_Knock_down
print(average_Knock_down_per_div)

# Compute the means of different statistics per fighter and division
average_stats_per_fighter_div = df_mma1.groupby(['fighter', 'division']).agg(
    {'total_strikes_attempts': 'mean', 'control': 'mean', 'knockdowns': 'mean', 'sub_attempts': 'mean'})
# Compute the means of different statistics per division
average_stats_per_div = df_mma1.groupby(['division']).agg(
    {'total_strikes_attempts': 'mean', 'control': 'mean', 'knockdowns': 'mean', 'sub_attempts': 'mean'})


# Define a function to classify each fighter into one of five fighter types
# Create an empty list to store the fighter types
fighter_types = []

# Iterate over each row of the DataFrame
for index, row in average_stats_per_fighter_div.iterrows():

    # Get the statistics for the current fighter and division
    strikes = row['total_strikes_attempts']
    control = row['control']
    knockdowns = row['knockdowns']
    subs = row['sub_attempts']

    # Get the average statistics for the fighter's division
    div = row.name[1]
    avg_strikes = average_stats_per_div.loc[div, 'total_strikes_attempts']
    avg_control = average_stats_per_div.loc[div, 'control']
    avg_knockdowns = average_stats_per_div.loc[div, 'knockdowns']
    avg_subs = average_stats_per_div.loc[div, 'sub_attempts']

    # Apply the conditions to classify the fighter into a fighter type
    if strikes > avg_strikes:
        if control > avg_control and knockdowns > avg_knockdowns and subs > avg_subs:
            fighter_types.append('Hybrid_Finisher')
        else:
            fighter_types.append('Volume_Striker')
    else:
        if knockdowns > avg_knockdowns and control < avg_control and subs < avg_subs:
            fighter_types.append('Knockout_Artist')
        elif subs > avg_subs:
            fighter_types.append('Submission_Specialist')
        else:
            fighter_types.append('Controller_Wrestler')

# Add the list of fighter types as a new column to the original DataFrame
average_stats_per_fighter_div['fighter_type'] = fighter_types
# merge the average_stats_per_fighter_div dataframe with df_mma1
df_mma2 = pd.merge(df_mma1, average_stats_per_fighter_div, on=['fighter', 'division'])
df_mma2=df_mma2.drop(columns=['total_strikes_attempts_y', 'control_y', 'knockdowns_y','sub_attempts_y'])
# Check repeated names
repeated_names = df_mma2[df_mma2.duplicated('fighter')]['fighter'].unique()

# Find fighters with different fighter types they are fighting
fighters_with_different_types = df_mma2.groupby('fighter')['fighter_type'].nunique()
fighters_with_different_types = fighters_with_different_types[fighters_with_different_types > 1].index.tolist()

# Filter out fighters who have only one fighter type
filtered_fighters = df_mma2.groupby('fighter').filter(lambda x: x['fighter_type'].nunique() > 1)['fighter'].unique()

# For each multi-fighter type fighter, find the most common fighter type
most_common_types = {}
for fighter in filtered_fighters:
    types = df_mma2[df_mma2['fighter'] == fighter]['fighter_type']
    most_common_type = types.value_counts().idxmax()
    most_common_types[fighter] = most_common_type

# Replace uncommon fighter types with the common fighter types
for fighter, common_type in most_common_types.items():
    df_mma2.loc[df_mma2['fighter'] == fighter, 'fighter_type'] = common_type

df_mma2.to_csv('updated_fighters1.csv', index=False)
############ EDA ##################################
# via tableau







