import pandas as pd
from math import pow

# Load dataset
df = pd.read_csv('updated_fights_V4_data.csv')

# Define function to get last rating of a fighter
def get_last_rating(fighter_name, rating_type):
    # Filter dataset to include only the fighter's name
    fighter_df = df[(df["fighter"] == fighter_name) | (df["opponent"] == fighter_name)]
    if fighter_df.empty:
        return None
    # Sort by date
    sorted_df = fighter_df.sort_values(by=["date"], ascending=False)
    # Return the last recorded rating of the specified type
    return sorted_df.iloc[0][rating_type]

# Example fighters
fighter1 = "Khabib Nurmagomedov"
fighter2 = "Conor McGregor"

# Get last recorded ratings of both fighters
fighter1_elo_rating = get_last_rating(fighter1, "elo_rating")
fighter2_elo_rating = get_last_rating(fighter2, "elo_rating")
fighter1_glicko_rating = get_last_rating(fighter1, "rating")
fighter2_glicko_rating = get_last_rating(fighter2, "rating")

# Check if data exists for both fighters
if fighter1_elo_rating is None:
    print(f"No data available for {fighter1}.")
if fighter2_elo_rating is None:
    print(f"No data available for {fighter2}.")
if fighter1_elo_rating is None or fighter2_elo_rating is None:
    exit()

# Calculate Elo rating difference
elo_diff = fighter2_elo_rating - fighter1_elo_rating

# Calculate win probability of fighter1 using Elo rating system
elo_fight_prob = 1 / (1 + pow(10, elo_diff / 400))
elo_oppo_prob=1-elo_fight_prob

print(f"Last recorded Elo rating for {fighter1}: {fighter1_elo_rating}")
print(f"Last recorded Elo rating for {fighter2}: {fighter2_elo_rating}")
print(f"Last recorded Glicko rating for {fighter1}: {fighter1_glicko_rating}")
print(f"Last recorded Glicko rating for {fighter2}: {fighter2_glicko_rating}")
print(f"Win probability of {fighter1} : {elo_fight_prob:.2%}")
print(f"Win probability of {fighter2} : {1 - elo_fight_prob:.2%}")

# Define the names of the fighter and opponent you want to find
fighter1 = 'Conor McGregor'
fighter2 = 'Dustin Poirier'

# Find the row(s) where fighter_name matches the fighter_name column
fighter_matches = df.loc[df['fighter'] == fighter1]

# Find the row(s) where opponent_name matches the fighter_name column
opponent_matches = df.loc[df['fighter'] == fighter2]

# If there are no matches for either fighter or opponent, print an error message
if fighter_matches.empty:
    print(f"No matches found for {fighter1}")
if opponent_matches.empty:
    print(f"No matches found for {fighter2}")

# Retrieve the last fighter_type value for the fighter
fighter_type = fighter_matches.iloc[[-1]]['fighter_type'].values[0]

# Retrieve the last fighter_type value for the opponent
opponent_type = opponent_matches.iloc[[-1]]['fighter_type'].values[0]

# Print the fighter_type values
print(fighter1 + "'s fighter_type is: " + fighter_type)
print(fighter2 + "'s fighter_type is: " + opponent_type)





# adding or subtracting percentage from the found rating

# volumne striker vs controller/wrestler +5 % probaility
# volumne striker vs ko artist -5 % probaility
# volumne striker vs hybrid finisher -5 % probaility
# volumne striker vs volumne striker no change in % probaility
# volumne striker vs  submission specailist no change % probaility
#
# controller/wrestler vs volumne striker -5% probaility
# controller/wrestler vs ko artist +5 % probaility
# controller/wrestler vs hybrid finisher -5 % probaility
# controller/wrestler  vs  submission specailist -5% probaility
# controller/wrestler vs controller/wrestler no change probaility
#
# hybrid finisher vs controller/wrestler +5 % probaility
# hybrid finisher vs ko artist no change probaility
# hybrid finisher vs hybrid finisher no change probaility
# hybrid finisher vs volumne striker +5% probaility
# hybrid finisher vs  submission specailist no change % probaility
#
# ko artist vs controller/wrestler -5 % probaility
# ko artist vs ko artist no change probaility
# ko artist vs hybrid finisher no change probaility
# ko artist vs volumne striker +5% probaility
# ko artist vs  submission specailist +5 % probaility
#
# submission specailist vs controller/wrestler +5 % probaility
# submission specailist vs ko artist -5% probaility
# submission specailist vs hybrid finisher no change probaility
# submission specailist vs volumne striker no change probaility
# submission specailist vs  submission specailist no change % probaility








