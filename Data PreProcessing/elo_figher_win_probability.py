import pandas as pd
import numpy as np
# from glicko2 import Glicko2
from datetime import datetime, timedelta

data=pd.read_csv('updated_fighters4.csv')



#P(fighter A wins) = 1 / (1 + 10^((rating of fighter B - rating of fighter A) / 400))
# E = 1 / (1 + 10^((Rb - Ra)/400))
# Ra is the fighter's current rating
# Rb is the opponent's rating
# ^ represents exponentiation

# fighters rating
# Rn = Ra + K * (S - E)
# Rn is the new rating
# K is the K-factor
# S is the actual score (1 for a win, 0 for a loss)
# E is the expected score


# Record: 20-5
# Elo Rating: 1900


fighter_Total_wins=data[data['result']==1].groupby('fighter')['fighter'].count()
fighter_total_loss=data[data['result']==0].groupby('fighter')['fighter'].count()

# Create a new column 'total_cumulative_wins' with the initial value of 0
data['total_cumulative_wins'] = 0

# Use groupby and transform to compute the total number of wins for each fighter
fighter_wins = data.groupby('fighter')['result'].transform(lambda x: x.eq(1).cumsum())

# Add the total wins for each fight to the 'total_wins' column for the winning fighter
data.loc[data['result'] == 1, 'total_cumulative_wins'] = fighter_wins

# data['total_cumulative_wins'] = data.groupby('fighter')['total_cumulative_wins'].apply(lambda x: x.replace(0, method='ffill'))
data['total_cumulative_wins'] = data.groupby('fighter', group_keys=False)['total_cumulative_wins'].apply(lambda x: x.replace(0, method='ffill'))

# elo rating
initial_elo = 1500

# Set the K-factor
k_factor = 32

elo_ratings = {}

# Iterate over each row in the dataset
for index, row in data.iterrows():
    # Get the fighter and opponent names
    fighter = row['fighter']
    opponent = row['opponent']

    # If the fighter is not in the Elo ratings dictionary, set their initial Elo rating
    if fighter not in elo_ratings:
        elo_ratings[fighter] = initial_elo

    # If the opponent is not in the Elo ratings dictionary, set their initial Elo rating
    if opponent not in elo_ratings:
        elo_ratings[opponent] = initial_elo

    # Calculate the expected probability of winning for the fighter
    expected = 1 / (1 + 10 ** ((elo_ratings[opponent] - elo_ratings[fighter]) / 400))

    # Calculate the actual result of the fight
    if row['result'] == 1:
        result = 1
    elif row['result'] == 0:
        result = 0
    else:
        result = 0.5

    # Calculate the new Elo rating for the fighter
    new_elo = elo_ratings[fighter] + k_factor * (result - expected)

    # Update the Elo ratings dictionary with the new rating for the fighter
    elo_ratings[fighter] = new_elo

# Create a new column in the dataset to store the Elo rating for each fighter
data['elo_rating'] = data['fighter'].apply(lambda x: elo_ratings[x])
data['elo_win_probability']=data.apply(lambda row: 1 / (1 + 10 ** ((elo_ratings[row['opponent']] - elo_ratings[row['fighter']]) / 400)), axis=1)

data.to_csv('updated_fighters333.csv', index=False)