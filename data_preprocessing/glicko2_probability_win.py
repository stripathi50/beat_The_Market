import pandas as pd
import math

# Define the initial rating values, deviations, and volatilities for each fighter
initial_rating = 1500
initial_deviation = 200
initial_volatility = 0.06

# Load the dataset
fights=pd.read_csv('updated_fighters2.csv')

# Define the Glicko-2 function
def glicko2_function(rd):
    q = math.log(10) / 400
    return 1 / math.sqrt(1 + 3 * q * q * rd * rd / math.pi / math.pi)

# Define the expected outcome function
def expected_outcome(r1, r2, rd2):
    q = math.log(10) / 400
    return 1 / (1 + math.exp(-glicko2_function(rd2) * q * (r1 - r2)))

# Define the update function
def update_ratings(r1, rd1, vol1, r2, rd2, vol2, outcome):
    q = math.log(10) / 400
    expected1 = expected_outcome(r1, r2, rd2)
    expected2 = 1 - expected1
    delta = glicko2_function(rd2) * q * (outcome - expected2)
    sigma = math.sqrt(vol1 * vol1 + vol2 * vol2)
    new_rd = math.sqrt(rd1 * rd1 + sigma * sigma)
    new_vol = vol1
    new_r = r1 + q / (1 / (rd1 * rd1) + 1 / new_rd / new_rd) * delta
    return new_r, new_rd, new_vol

# Define a function to update the ratings for each fighter after each fight
def update_fighter_ratings(fights):
    ratings = {fighter: {'rating': initial_rating, 'deviation': initial_deviation, 'volatility': initial_volatility} for fighter in fights['fighter'].unique()}
    new_ratings = []
    for i, row in fights.iterrows():
        fighter1 = row['fighter']
        fighter2 = row['opponent']
        outcome = row['result']
        r1, rd1, vol1 = ratings[fighter1]['rating'], ratings[fighter1]['deviation'], ratings[fighter1]['volatility']
        r2, rd2, vol2 = ratings[fighter2]['rating'], ratings[fighter2]['deviation'], ratings[fighter2]['volatility']
        new_r1, new_rd1, new_vol1 = update_ratings(r1, rd1, vol1, r2, rd2, vol2, outcome)
        new_r2, new_rd2, new_vol2 = update_ratings(r2, rd2, vol2, r1, rd1, vol1, 1 - outcome)
        ratings[fighter1]['rating'], ratings[fighter1]['deviation'], ratings[fighter1]['volatility'] = new_r1, new_rd1, new_vol1
        ratings[fighter2]['rating'], ratings[fighter2]['deviation'], ratings[fighter2]['volatility'] = new_r2, new_rd2, new_vol2
        glicko2_win_probability = expected_outcome(r1, r2, rd2)
        new_ratings.append(
            {'fighter': fighter1, 'opponent': fighter2, 'result': outcome, 'rating': new_r1, 'deviation': new_rd1,
             'volatility': new_vol1, 'glicko2_win_probability': glicko2_win_probability})
        new_ratings.append(
            {'fighter': fighter2, 'opponent': fighter1, 'result': 1 - outcome, 'rating': new_r2, 'deviation': new_rd2,
             'volatility': new_vol2, 'glicko2_win_probability': 1 - glicko2_win_probability})

    return pd.DataFrame(new_ratings)

updated_fights = update_fighter_ratings(fights)
updated_fights = updated_fights.drop_duplicates(['fighter', 'opponent', 'result'])

data=pd.read_csv('updated_fighters2.csv')
merge_data=data.merge(updated_fights, on=['fighter', 'opponent', 'result'], how='left')
merge_data.to_csv('updated_fighters3.csv', index=False)

