import math
import pandas as pd

# Initial values for new fighters
initial_rating = 1500
initial_deviation = 200
initial_volatility = 0.06

def g(r_d):
    return 1 / math.sqrt(1 + 3 * math.pow(r_d, 2) / math.pow(math.pi, 2))

def E(r_i, r_j, rd_j):
    return 1 / (1 + math.exp(-g(rd_j) * (r_i - r_j)))

def d_squared(r_i, r_j, rd_j, results):
    return 1 / (math.pow(g(rd_j), 2) * sum([math.pow(g(rd_j) * (result - E(r_i, r_j, rd_j)), 2) for result in results]))

def new_deviation(r_i, r_j, rd_i, rd_j, results):
    dsq = d_squared(r_i, r_j, rd_j, results)
    return 1 / math.sqrt(1 / math.pow(rd_i, 2) + 1 / dsq)

def new_volatility(r_i, rd_i, v, results):
    dsq = d_squared(r_i, r_i, rd_i, results)
    rd_new = new_deviation(r_i, r_i, rd_i, rd_i, results)
    return math.sqrt(math.pow(rd_new, 2) - math.pow(rd_i, 2))

def update_rating(r_i, rd_i, v_i, r_j, rd_j, v_j, result):
    q = math.log(10) / 400
    E_i = E(r_i, r_j, rd_j)
    d2 = d_squared(r_i, r_j, rd_j, [result])
    new_r_i = r_i + q / (1 / math.pow(rd_i, 2) + 1 / (g(rd_j) * g(rd_j) * d2)) * g(rd_j) * (result - E_i)
    new_rd_i = new_deviation(r_i, r_j, rd_i, rd_j, [result])
    new_v_i = new_volatility(r_i, rd_i, v_i, [result])
    return new_r_i, new_rd_i, new_v_i

def calculate_glicko2_win_probability(fighter1, fighter2, ratings):
    if fighter1 in ratings.fighter and fighter2 in ratings.fighter:
        r_i, rd_i, v_i = ratings.loc[fighter1].iloc[-1][['rating', 'deviation', 'volatility']]
        print(r_i, rd_i, v_i)
        r_j, rd_j, v_j = ratings.loc[fighter2].iloc[-1][['rating', 'deviation', 'volatility']]
        print(r_j, rd_j, v_j)
        return E(r_i, r_j, rd_j)
    elif fighter1 in ratings.fighter:
        r_i, rd_i, v_i = ratings.loc[fighter1].iloc[-1][['rating', 'deviation', 'volatility']]
        return E(r_i, initial_rating, initial_deviation)
    elif fighter2 in ratings.fighter:
        r_j, rd_j, v_j = ratings.loc[fighter2].iloc[-1][['rating', 'deviation', 'volatility']]
        return E(initial_rating, r_j, rd_j)
    else:
        return E(initial_rating, initial_rating, initial_deviation)

# Example usage
# Assume that 'ratings' is a pandas DataFrame with columns 'fighter', 'rating', 'deviation', and 'volatility'
fighter1 = 'Conor McGregor'
fighter2 = 'Khabib Nurmagomedov'
data = pd.read_csv('updated_fights_V4_data.csv')

win_probability = calculate_glicko2_win_probability(fighter1,fighter2,data)
print(f"Glicko2 win probability for {fighter1} vs {fighter2}: {win_probability:.2%}")
