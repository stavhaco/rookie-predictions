import requests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nba_py import player
from nba_py.player import get_player
from sklearn.model_selection import train_test_split

def rookies_by_year(start_year=1997,end_year=2010):
    frames = []
    for year in range(start_year,end_year,1):
        headers_mozila = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/61.0.3163.100 Safari/537.36'}
        next_year = year + 1
        year_string = str(year) + '-' + str(next_year)[2] + str(next_year)[3] #format year yyyy-yy
        print year_string
        rookie_year_url = 'https://stats.nba.com/stats/leaguedashplayerstats?College=&Conference=&Country=&DateFrom=&DateTo' \
                         '=&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location' \
                         '=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period' \
                         '=0&PlayerExperience=Rookie&PlayerPosition=&PlusMinus=N&Rank=N&Season='+year_string+'&SeasonSegment' \
                         '=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&VsConference=&VsDivision=&Weight= '
        response_playoffs = requests.get(rookie_year_url, headers=headers_mozila)
        col_headers_playoffs = response_playoffs.json()['resultSets'][0]['headers']
        team_stats_playoffs = response_playoffs.json()['resultSets'][0]['rowSet']
        team_stats_df = pd.DataFrame(team_stats_playoffs, columns=col_headers_playoffs)
        frames.append(team_stats_df)
    return pd.concat(frames)
#s
all_rookies = rookies_by_year()

train, test = train_test_split(all_rookies, test_size=0.2)
train.to_csv(path_or_buf='C:/Users/stav/Desktop/study/nba/rookies_train.csv')
test.to_csv(path_or_buf='C:/Users/stav/Desktop/study/nba/rookies_test.csv')

