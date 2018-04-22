import pandas as pd
from nba_py import player
import numpy as np
from nba_py.player import get_player, PlayerProfile




def best_EFF():
    rookie_data = pd.DataFrame.from_csv('C:/Users/stav/Desktop/study/nba/rookies_train.csv')
    rookie_data.index = range(len(rookie_data))
    rookie_ids = rookie_data['PLAYER_ID']
    row = 0
    for pid in rookie_ids:
        print row, pid, "row pid"
        rank_EFF_by_year = player.PlayerCareer(pid).regular_season_rankings()["RANK_PG_EFF"]
        #if player.PlayerCareer(pid).regular_season_career_totals()["GP"][0]<10:
        #    best_pid_EFF=None
        if all(r is None for r in rank_EFF_by_year):
            best_pid_EFF=None
        else:
            best_pid_EFF = np.nanmin(rank_EFF_by_year.iloc[:].values)
        #print "best eff "+str(best_pid_EFF)
        rookie_data.at[row,"BEST EFF"]=best_pid_EFF
        row += 1
        if row ==10:
            print rookie_data
    return rookie_data


#print range


df = best_EFF()

df.to_csv(path_or_buf='C:/Users/stav/Desktop/study/nba/rookies_train_EFF_rank.csv')



