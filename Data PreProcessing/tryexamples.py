import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression



# columns_to_keep = ['R_fighter','B_fighter','R_odds','B_odds','country','Winner','weight_class',
#                    'gender','no_of_rounds','B_age','R_age','R_Reach_cms','R_Height_cms','R_Stance',
#                    'R_wins','reversals','control','takedowns_landed','takedowns_attempts',
#                    'sig_strikes_landed','sig_strikes_attempts','total_strikes_attempts',
#                    'sig_strikes_landed','avg_control','age_differential','avg_knockdowns',
#                    'avg_sub_attempts','avg_reversals','avg_control','avg_takedowns_attempts',
#                    'avg_sig_strikes_landed','avg_sig_strikes_attempts','avg_total_strikes_landed',
#                    'avg_total_strikes_attempts']

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df_mma1=pd.read_csv('/Users/sagartripathi/Documents/pythonProject/MMA_Betting/data/masterdataframe.csv')

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

df_mma1 = df_mma1.drop(index=range(1000))
# impute the height values for Edward Faaloloto and Tom Blackledge
height_values = {'Edward Faaloloto': 69, 'Tom Blackledge': 72}


for name, height in height_values.items():
    df_mma1.loc[df_mma1['fighter'] == name, 'height'] = height

df_mma1['reach'].fillna(df_mma1['height'], inplace=True)

print(df_mma1.isnull().sum().sort_values())



print(df_mma1.head(5))


# finding_oponent =
# ['Rick Davis', 'Cory Walmsley', 'David Lee', 'Mario Neto', 'Steve Byrnes', 'Victor Valimaki', 'Jason Gilliam', 'Rex Holman', 'Jess Liaudin', 'Victor Valimaki', 'David Lee', 'Stevie Lynch', 'Jason Gilliam', 'Jess Liaudin', 'Per Eklund', 'Jess Liaudin', 'Jess Liaudin', 'Michael Patt', 'Joe Vedepo', 'Per Eklund', 'Jess Liaudin', 'Neil Wain', 'Ivan Serati', 'Tom Egan', 'Per Eklund', 'Brian Cobb', 'Michael Patt', 'Ryan Madigan', 'Joe Vedepo', 'Jesse Sanders', 'Jay Silva', 'Chase Gormley', 'Jay Silva', 'Chase Gormley', 'Curt Warburton', 'Mark Scanlon', 'Curt Warburton', 'Tom Blackledge', 'Curt Warburton', 'Matt Lucas']
# replacing null value of dob with dob of the opponent.
null_dob_records = df_mma1[df_mma1['dob'].isnull()]
y=null_dob_records[['fighter','opponent']]
records_for_dob=[]
for i,j in y.values:
    records_for_dob.append(df_mma1[(df_mma1['fighter']==j )& (df_mma1['opponent']==i)]['dob'])
df_mma1.loc[df_mma1['dob'].isnull(), 'dob'] = records_for_dob


print(df_mma1.isnull().sum().sort_values())

#Rick 1983-03-30 1983-03-30
#1976-02-10 1976-02-10




