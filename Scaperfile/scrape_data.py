import requests
import urllib3
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime, timedelta
pd.set_option('display.max_columns', 500)

from ufc_scraper import get_fight_stats, get_fight_card, get_all_fight_stats,\
    get_fighter_details, update_fight_stats, update_fighter_details

#scrape data from scratch
fight_hist = get_all_fight_stats()

fighter_stats = get_fighter_details(fight_hist.fighter_url.unique())

#save data
fight_hist.to_csv('/Users/sagartripathi/Documents/pythonProject/MMA_Betting/data/fight_hist.csv', index = False)
fighter_stats.to_csv('/Users/sagartripathi/Documents/pythonProject/MMA_Betting/data/fighter_stats.csv', index = False)

# get already saved data
fight_hist_old = pd.read_csv('/Users/sagartripathi/Documents/pythonProject/MMA_Betting/data/fight_hist.csv')

fighter_stats_old = pd.read_csv('/Users/sagartripathi/Documents/pythonProject/MMA_Betting/data/fighter_stats.csv')

# update fight hist dataframe
fight_hist_updated = update_fight_stats(fight_hist_old)

# update fighter stats
fighter_stats_updated = update_fighter_details(fight_hist_updated.fighter_url.unique(), fighter_stats_old)

#save updated dataframes
fight_hist_updated.to_csv('/Users/sagartripathi/Documents/pythonProject/MMA_Betting/data/fight_hist.csv', index = False)
fighter_stats_updated.to_csv('/Users/sagartripathi/Documents/pythonProject/MMA_Betting/data/fighter_stats.csv', index = False)

print(fight_hist_updated)
