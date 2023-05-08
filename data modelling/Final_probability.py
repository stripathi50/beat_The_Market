import pandas as pd
from math import pow
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('updated_fighters3.csv')

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

# # Example fighters
# fighter1 = "Caio Borralho"
# fighter2 = "Michal Oleksiejczuk"
#
# # Get last recorded ratings of both fighters
# fighter1_elo_rating = get_last_rating(fighter1, "elo_rating")
# fighter2_elo_rating = get_last_rating(fighter2, "elo_rating")
# fighter1_glicko_rating = get_last_rating(fighter1, "rating")
# fighter2_glicko_rating = get_last_rating(fighter2, "rating")
#
# # Check if data exists for both fighters
# if fighter1_elo_rating is None:
#     print(f"No data available for {fighter1}.")
# if fighter2_elo_rating is None:
#     print(f"No data available for {fighter2}.")
# if fighter1_elo_rating is None or fighter2_elo_rating is None:
#     exit()
#
# # Calculate Elo rating difference
# elo_diff = fighter2_elo_rating - fighter1_elo_rating
# x=1 + 10**(elo_diff / 400)
# # Calculate win probability of fighter1 using Elo rating system
# elo_fight_prob = 1 /x
# elo_oppo_prob=1-elo_fight_prob
#
# print(f"Last recorded Elo rating for {fighter1}: {fighter1_elo_rating}")
# print(f"Last recorded Elo rating for {fighter2}: {fighter2_elo_rating}")
# print(f"Last recorded Glicko rating for {fighter1}: {fighter1_glicko_rating}")
# print(f"Last recorded Glicko rating for {fighter2}: {fighter2_glicko_rating}")
# print(f"Win probability of {fighter1} : {elo_fight_prob:.2%}")
# print(f"Win probability of {fighter2} : {1 - elo_fight_prob:.2%}")
#
# # Define the names of the fighter and opponent you want to find
#
# # Find the row(s) where fighter_name matches the fighter_name column
# fighter_matches = df.loc[df['fighter'] == fighter1]
#
# # Find the row(s) where opponent_name matches the fighter_name column
# opponent_matches = df.loc[df['fighter'] == fighter2]
#
# # If there are no matches for either fighter or opponent, print an error message
# if fighter_matches.empty:
#     print(f"No matches found for {fighter1}")
# if opponent_matches.empty:
#     print(f"No matches found for {fighter2}")
#
# # Retrieve the last fighter_type value for the fighter
# fighter_type = fighter_matches.iloc[[-1]]['fighter_type'].values[0]
#
# # Retrieve the last fighter_type value for the opponent
# opponent_type = opponent_matches.iloc[[-1]]['fighter_type'].values[0]
#
# #IF FIGHTER ONE FIGHTER TYPE IS AND OPPONENT FIGHTER TYPE IS
#
# # adding or subtracting percentage from the found rating
#
# # volumne striker vs controller/wrestler +5 % probaility
# # volumne striker vs ko artist -5 % probaility
# # volumne striker vs hybrid finisher -5 % probaility
# # volumne striker vs volumne striker no change in % probaility
# # volumne striker vs  submission specailist no change % probaility
# #
# # controller/wrestler vs volumne striker -5% probaility
# # controller/wrestler vs ko artist +5 % probaility
# # controller/wrestler vs hybrid finisher -5 % probaility
# # controller/wrestler  vs  submission specailist -5% probaility
# # controller/wrestler vs controller/wrestler no change probaility
# #
# # hybrid finisher vs controller/wrestler +5 % probaility
# # hybrid finisher vs ko artist no change probaility
# # hybrid finisher vs hybrid finisher no change probaility
# # hybrid finisher vs volumne striker +5% probaility
# # hybrid finisher vs  submission specailist no change % probaility
# #
# # ko artist vs controller/wrestler -5 % probaility
# # ko artist vs ko artist no change probaility
# # ko artist vs hybrid finisher no change probaility
# # ko artist vs volumne striker +5% probaility
# # ko artist vs  submission specailist +5 % probaility
# #
# # submission specailist vs controller/wrestler +5 % probaility
# # submission specailist vs ko artist -5% probaility
# # submission specailist vs hybrid finisher no change probaility
# # submission specailist vs volumne striker no change probaility
# # submission specailist vs  submission specailist no change % probaility
#
#
#
# def calculate_probability(fighter_one_type, fighter_two_type, elo_fight_prob, elo_oppo_prob):
#     if fighter_one_type == "Volume_Striker":
#         if fighter_two_type == "Controller_Wrestler":
#             x = elo_fight_prob + 0.05
#             y = elo_oppo_prob - 0.05
#             return x, y
#         elif fighter_two_type == "Knockout_Artist":
#             x = elo_fight_prob - 0.05
#             y = elo_oppo_prob + 0.05
#             return x, y
#         elif fighter_two_type == "Hybrid_Finisher":
#             x = elo_fight_prob - 0.05
#             y = elo_oppo_prob
#             return x, y
#         elif fighter_two_type == "Volume_Striker":
#             x = elo_fight_prob
#             y = elo_oppo_prob
#             return x, y
#         elif fighter_two_type == "Submission_Specialist":
#             x = elo_fight_prob
#             y = elo_oppo_prob
#             return x, y
#     elif fighter_one_type == "Controller_Wrestler":
#         if fighter_two_type == "Volume_Striker":
#             x = elo_fight_prob - 0.05
#             y = elo_oppo_prob + 0.05
#             return x, y
#         elif fighter_two_type == "Knockout_Artist":
#             x = elo_fight_prob + 0.05
#             y = elo_oppo_prob - 0.05
#             return x, y
#         elif fighter_two_type == "Hybrid_Finisher":
#             x = elo_fight_prob - 0.05
#             y = elo_oppo_prob
#             return x, y
#         elif fighter_two_type == "Submission_Specialist":
#             x = elo_fight_prob - 0.05
#             y = elo_oppo_prob
#             return x, y
#         elif fighter_two_type == "Controller_Wrestler":
#             x = elo_fight_prob
#             y = elo_oppo_prob
#             return x, y
#     elif fighter_one_type == "hybrid finisher":
#         if fighter_two_type == "Controller_Wrestler":
#             x = elo_fight_prob + 0.05
#             y = elo_oppo_prob - 0.05
#             return x, y
#         elif fighter_two_type == "Knockout_Artist":
#             x = elo_fight_prob
#             y = elo_oppo_prob
#             return x, y
#         elif fighter_two_type == "Hybrid_Finisher":
#             x = elo_fight_prob
#             y = elo_oppo_prob
#             return x, y
#         elif fighter_two_type == "Volume_Striker":
#             x = elo_fight_prob + 0.05
#             y = elo_oppo_prob
#             return x, y
#         elif fighter_two_type == "Submission_Specialist":
#             x = elo_fight_prob
#             y = elo_oppo_prob
#             return x, y
#     elif fighter_one_type == "Knockout_Artist":
#         if fighter_two_type == "Controller_Wrestler":
#             x = elo_fight_prob - 0.05
#             y = elo_oppo_prob + 0.05
#             return x, y
#         elif fighter_two_type == "Hybrid_Finisher":
#             x = elo_fight_prob
#             y = elo_oppo_prob
#             return x, y
#         elif fighter_two_type == "Volume_Striker":
#             x = elo_fight_prob + 0.05
#             y= elo_oppo_prob - 0.05
#             return x,y
#         elif fighter_two_type == 'Submission_Specialist':
#             x = elo_fight_prob + 0.05
#             y= elo_oppo_prob - 0.05
#             return x,y
#         elif fighter_two_type == 'Knockout_Artist':
#             x = elo_fight_prob
#             y = elo_oppo_prob
#             return x, y
#     elif fighter_one_type == "Submission_Specialist":
#         if fighter_two_type == "Controller_Wrestler":
#             x = elo_fight_prob + 0.05
#             y = elo_oppo_prob - 0.05
#             return x, y
#         elif fighter_two_type == "Hybrid_Finisher":
#             x = elo_fight_prob
#             y = elo_oppo_prob
#             return x, y
#         elif fighter_two_type == "Volume_Striker":
#             x = elo_fight_prob
#             y = elo_oppo_prob
#             return x, y
#         elif fighter_two_type == 'Submission_Specialist':
#             x = elo_fight_prob
#             y = elo_oppo_prob
#             return x, y
#         elif fighter_two_type == 'Knockout_Artist':
#             x = elo_fight_prob - 0.05
#             y = elo_oppo_prob + 0.05
#             return x, y
#
#
# x,y=calculate_probability(fighter_type,opponent_type,elo_fight_prob,elo_oppo_prob)
# print(fighter1 + "'s fighter_type is: " + fighter_type)
# print(fighter2 + "'s fighter_type is: " + opponent_type)
#
# # Implied win probabilities for 'Aljamain Sterling' and 'Henry Cejudo'
# implied_prob_a = x
# implied_prob_b = y
#
# print(f"New Win probability of {fighter1} : {x:.2%}")
# print(f"New Win probability of {fighter2} : {y:.2%}")
#
#
#
# # Calculate decimal odds
# decimal_odds_a = 1 / implied_prob_a
# decimal_odds_b = 1 / implied_prob_b
# # Print the results
# print(f"Decimal Odds for {fighter1}:", decimal_odds_a)
# print(f"Decimal Odds for {fighter2}:", decimal_odds_b)
#
# # Data
# percentages = [x, y]
# labels = [f'{fighter1} \n Odds:{round(decimal_odds_a,2)}',f'{fighter2}\n Odds:{round(decimal_odds_b,2)}']
# colors = ['blue', 'red']
#
# # Plot
# plt.pie(percentages, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
#
# # Aspect ratio to make the pie circular
# plt.axis('equal')
#
# # Title
# plt.title(f'{fighter1} vs {fighter2}')
# plt.tight_layout()
#
# # Display the chart
# plt.show()


import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

app = dash.Dash(__name__)
fighters=df['fighter'].unique()
app.layout = html.Div([
    html.H1('Fighter winning odds Comparison'),
    html.Div([
        html.Label('Select Fighter 1:'),
        dcc.Dropdown(
            id='fighter1-dropdown',
            options=[{'label': fighter, 'value': fighter} for fighter in fighters],
            value=''
        )
    ]),
    html.Div([
        html.Label('Select Fighter 2:'),
        dcc.Dropdown(
            id='fighter2-dropdown',
            options=[{'label': fighter, 'value': fighter} for fighter in fighters],
            value=''
        )
    ]),
    html.Div(id='output'),
    dcc.Graph(id='graph')
])

@app.callback(
    dash.dependencies.Output('output', 'children'),
    [dash.dependencies.Input('fighter1-dropdown', 'value'),
     dash.dependencies.Input('fighter2-dropdown', 'value')]
)
def update_output(fighter1, fighter2):
    # Example fighters

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
    x = 1 + 10 ** (elo_diff / 400)
    # Calculate win probability of fighter1 using Elo rating system
    elo_fight_prob = 1 / x
    elo_oppo_prob = 1 - elo_fight_prob

    # Define the names of the fighter and opponent you want to find

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

    def calculate_probability(fighter_one_type, fighter_two_type, elo_fight_prob, elo_oppo_prob):
        if fighter_one_type == "Volume_Striker":
            if fighter_two_type == "Controller_Wrestler":
                x = elo_fight_prob + 0.05
                y = elo_oppo_prob - 0.05
                return x, y
            elif fighter_two_type == "Knockout_Artist":
                x = elo_fight_prob - 0.05
                y = elo_oppo_prob + 0.05
                return x, y
            elif fighter_two_type == "Hybrid_Finisher":
                x = elo_fight_prob - 0.05
                y = elo_oppo_prob
                return x, y
            elif fighter_two_type == "Volume_Striker":
                x = elo_fight_prob
                y = elo_oppo_prob
                return x, y
            elif fighter_two_type == "Submission_Specialist":
                x = elo_fight_prob
                y = elo_oppo_prob
                return x, y
        elif fighter_one_type == "Controller_Wrestler":
            if fighter_two_type == "Volume_Striker":
                x = elo_fight_prob - 0.05
                y = elo_oppo_prob + 0.05
                return x, y
            elif fighter_two_type == "Knockout_Artist":
                x = elo_fight_prob + 0.05
                y = elo_oppo_prob - 0.05
                return x, y
            elif fighter_two_type == "Hybrid_Finisher":
                x = elo_fight_prob - 0.05
                y = elo_oppo_prob
                return x, y
            elif fighter_two_type == "Submission_Specialist":
                x = elo_fight_prob - 0.05
                y = elo_oppo_prob
                return x, y
            elif fighter_two_type == "Controller_Wrestler":
                x = elo_fight_prob
                y = elo_oppo_prob
                return x, y
        elif fighter_one_type == "hybrid finisher":
            if fighter_two_type == "Controller_Wrestler":
                x = elo_fight_prob + 0.05
                y = elo_oppo_prob - 0.05
                return x, y
            elif fighter_two_type == "Knockout_Artist":
                x = elo_fight_prob
                y = elo_oppo_prob
                return x, y
            elif fighter_two_type == "Hybrid_Finisher":
                x = elo_fight_prob
                y = elo_oppo_prob
                return x, y
            elif fighter_two_type == "Volume_Striker":
                x = elo_fight_prob + 0.05
                y = elo_oppo_prob
                return x, y
            elif fighter_two_type == "Submission_Specialist":
                x = elo_fight_prob
                y = elo_oppo_prob
                return x, y
        elif fighter_one_type == "Knockout_Artist":
            if fighter_two_type == "Controller_Wrestler":
                x = elo_fight_prob - 0.05
                y = elo_oppo_prob + 0.05
                return x, y
            elif fighter_two_type == "Hybrid_Finisher":
                x = elo_fight_prob
                y = elo_oppo_prob
                return x, y
            elif fighter_two_type == "Volume_Striker":
                x = elo_fight_prob + 0.05
                y = elo_oppo_prob - 0.05
                return x, y
            elif fighter_two_type == 'Submission_Specialist':
                x = elo_fight_prob + 0.05
                y = elo_oppo_prob - 0.05
                return x, y
            elif fighter_two_type == 'Knockout_Artist':
                x = elo_fight_prob
                y = elo_oppo_prob
                return x, y
        elif fighter_one_type == "Submission_Specialist":
            if fighter_two_type == "Controller_Wrestler":
                x = elo_fight_prob + 0.05
                y = elo_oppo_prob - 0.05
                return x, y
            elif fighter_two_type == "Hybrid_Finisher":
                x = elo_fight_prob
                y = elo_oppo_prob
                return x, y
            elif fighter_two_type == "Volume_Striker":
                x = elo_fight_prob
                y = elo_oppo_prob
                return x, y
            elif fighter_two_type == 'Submission_Specialist':
                x = elo_fight_prob
                y = elo_oppo_prob
                return x, y
            elif fighter_two_type == 'Knockout_Artist':
                x = elo_fight_prob - 0.05
                y = elo_oppo_prob + 0.05
                return x, y

    x, y = calculate_probability(fighter_type, opponent_type, elo_fight_prob, elo_oppo_prob)

    # Implied win probabilities for 'Aljamain Sterling' and 'Henry Cejudo'
    implied_prob_a = x
    implied_prob_b = y


    # Calculate decimal odds
    decimal_odds_a = 1 / implied_prob_a
    decimal_odds_b = 1 / implied_prob_b
    # Print the results
    if fighter1 and fighter2:
        return html.Div([
            html.H3(f"You selected: {fighter1} and {fighter2}"),
            html.P(f"{fighter1} win percentage: {x:.2%} and {fighter2} win percentage: {y:.2%}"),
            html.P(f"{fighter1} decimal odds: {decimal_odds_a} and {fighter2} decimal odds: {decimal_odds_b}")
        ])
    else:
        return ''


@app.callback(
    dash.dependencies.Output('graph', 'figure'),
    [dash.dependencies.Input('fighter1-dropdown', 'value'),
     dash.dependencies.Input('fighter2-dropdown', 'value')]
)
def update_graph(fighter1, fighter2):
    if fighter1 and fighter2:
        # Generate sample data for the pie chart
        # Sample data, replace with your actual data
        # Example fighters
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
        x = 1 + 10 ** (elo_diff / 400)
        # Calculate win probability of fighter1 using Elo rating system
        elo_fight_prob = 1 / x
        elo_oppo_prob = 1 - elo_fight_prob

        # Define the names of the fighter and opponent you want to find

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
        def calculate_probability(fighter_one_type, fighter_two_type, elo_fight_prob, elo_oppo_prob):
            if fighter_one_type == "Volume_Striker":
                if fighter_two_type == "Controller_Wrestler":
                    x = elo_fight_prob + 0.05
                    y = elo_oppo_prob - 0.05
                    return x, y
                elif fighter_two_type == "Knockout_Artist":
                    x = elo_fight_prob - 0.05
                    y = elo_oppo_prob + 0.05
                    return x, y
                elif fighter_two_type == "Hybrid_Finisher":
                    x = elo_fight_prob - 0.05
                    y = elo_oppo_prob
                    return x, y
                elif fighter_two_type == "Volume_Striker":
                    x = elo_fight_prob
                    y = elo_oppo_prob
                    return x, y
                elif fighter_two_type == "Submission_Specialist":
                    x = elo_fight_prob
                    y = elo_oppo_prob
                    return x, y
            elif fighter_one_type == "Controller_Wrestler":
                if fighter_two_type == "Volume_Striker":
                    x = elo_fight_prob - 0.05
                    y = elo_oppo_prob + 0.05
                    return x, y
                elif fighter_two_type == "Knockout_Artist":
                    x = elo_fight_prob + 0.05
                    y = elo_oppo_prob - 0.05
                    return x, y
                elif fighter_two_type == "Hybrid_Finisher":
                    x = elo_fight_prob - 0.05
                    y = elo_oppo_prob
                    return x, y
                elif fighter_two_type == "Submission_Specialist":
                    x = elo_fight_prob - 0.05
                    y = elo_oppo_prob
                    return x, y
                elif fighter_two_type == "Controller_Wrestler":
                    x = elo_fight_prob
                    y = elo_oppo_prob
                    return x, y
            elif fighter_one_type == "hybrid finisher":
                if fighter_two_type == "Controller_Wrestler":
                    x = elo_fight_prob + 0.05
                    y = elo_oppo_prob - 0.05
                    return x, y
                elif fighter_two_type == "Knockout_Artist":
                    x = elo_fight_prob
                    y = elo_oppo_prob
                    return x, y
                elif fighter_two_type == "Hybrid_Finisher":
                    x = elo_fight_prob
                    y = elo_oppo_prob
                    return x, y
                elif fighter_two_type == "Volume_Striker":
                    x = elo_fight_prob + 0.05
                    y = elo_oppo_prob
                    return x, y
                elif fighter_two_type == "Submission_Specialist":
                    x = elo_fight_prob
                    y = elo_oppo_prob
                    return x, y
            elif fighter_one_type == "Knockout_Artist":
                if fighter_two_type == "Controller_Wrestler":
                    x = elo_fight_prob - 0.05
                    y = elo_oppo_prob + 0.05
                    return x, y
                elif fighter_two_type == "Hybrid_Finisher":
                    x = elo_fight_prob
                    y = elo_oppo_prob
                    return x, y
                elif fighter_two_type == "Volume_Striker":
                    x = elo_fight_prob + 0.05
                    y = elo_oppo_prob - 0.05
                    return x, y
                elif fighter_two_type == 'Submission_Specialist':
                    x = elo_fight_prob + 0.05
                    y = elo_oppo_prob - 0.05
                    return x, y
                elif fighter_two_type == 'Knockout_Artist':
                    x = elo_fight_prob
                    y = elo_oppo_prob
                    return x, y
            elif fighter_one_type == "Submission_Specialist":
                if fighter_two_type == "Controller_Wrestler":
                    x = elo_fight_prob + 0.05
                    y = elo_oppo_prob - 0.05
                    return x, y
                elif fighter_two_type == "Hybrid_Finisher":
                    x = elo_fight_prob
                    y = elo_oppo_prob
                    return x, y
                elif fighter_two_type == "Volume_Striker":
                    x = elo_fight_prob
                    y = elo_oppo_prob
                    return x, y
                elif fighter_two_type == 'Submission_Specialist':
                    x = elo_fight_prob
                    y = elo_oppo_prob
                    return x, y
                elif fighter_two_type == 'Knockout_Artist':
                    x = elo_fight_prob - 0.05
                    y = elo_oppo_prob + 0.05
                    return x, y

        x, y = calculate_probability(fighter_type, opponent_type, elo_fight_prob, elo_oppo_prob)
        implied_prob_a = x
        implied_prob_b = y


        # Calculate decimal odds
        decimal_odds_a = 1 / implied_prob_a
        decimal_odds_b = 1 / implied_prob_b
        percentages = [x, y]
        labels = [f'{fighter1} \n Odds:{round(decimal_odds_a, 2)}', f'{fighter2}\n Odds:{round(decimal_odds_b, 2)}']
        colors = ['blue', 'red']

        fig = go.Figure(data=[go.Pie(labels=labels, values=percentages, marker=dict(colors=colors))])

        fig.update_layout(title='Fighter Comparison')

        return fig
    else:
        return {}


# if __name__ == '__main__':
#     app.run_server(debug=True)

if __name__=='__main__':

    app.run_server(
     port=8024,
     host='0.0.0.0')



