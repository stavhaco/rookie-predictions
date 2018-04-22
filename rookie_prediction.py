import pandas as pd

df = pd.read_csv("Seasons_Stats.csv")

def make_identifier(df):
    str_id = df.apply(lambda x: '_'.join(map(str, x)), axis=1)
    return pd.factorize(str_id)[0]

df['id'] = make_identifier(df[['Player','Pos']])
print df[df["Player"]=="Ray Allen"]
id_maxPER = df.groupby(['id'])['PER'].transform(max) == df['PER']
print df[id_maxPER]

df_modern = df[df["Year"]>1979]
df_sorted = df_modern.sort_values(by="Year")
df_rookie = df_sorted.drop_duplicates(subset=["Player"])




































