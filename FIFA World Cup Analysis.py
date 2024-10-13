#!/usr/bin/env python
# coding: utf-8

# # The FIFA World Cup
# The FIFA World Cup, often simply called the World Cup, is an international association football competition contested by the senior men's national teams of the members of the Fédération Internationale de Football Association (FIFA), the sport's global governing body. The championship has been awarded every four years since the inaugural tournament in 1930, except in 1942 and 1946 when it was not held because of the Second World War. The current champion is Germany, which won its fourth title at the 2014 tournament in Brazil.
# 
# ## Data - content 
# The World Cups dataset show all information about all the World Cups in the history, while the World Cup Matches dataset shows all the results from the matches contested as part of the cups.

# In[127]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import  Image
get_ipython().run_line_magic('matplotlib', 'inline')


# In[128]:


#importing libraries
import pandas as pd
import seaborn as sns
import itertools
import warnings
warnings.filterwarnings("ignore")
import io
import base64
from matplotlib import rc,animation
from mpl_toolkits.mplot3d import Axes3D
#import os
#print(os.listdir("../input"))


# # Data - Overview

# In[129]:


matches  = pd.read_csv(r"D:\Unified Mentor\FIFA WC data\WorldCupMatches.csv")
players  = pd.read_csv(r"D:\Unified Mentor\FIFA WC data\WorldCupPlayers.csv")
cups     = pd.read_csv(r"D:\Unified Mentor\FIFA WC data\WorldCups.csv")
display("MATCHES - DATA")
display(matches.head(3))
display("PLAYERS - DATA")
display(players.head(3))
display("WORLD CUPS - DATA")
display(cups.head(3))


# # Total attendence of world cups by year
# * The championship has been awarded every four years since the inaugural tournament in 1930, except in 1942 and 1946 when it was not held because of the Second World War.

# In[130]:


matches.isnull().sum()

sns.set_style("darkgrid")  
matches = matches.drop_duplicates(subset="MatchID", keep="first")
matches = matches[matches["Year"].notnull()]
att = matches.groupby("Year")["Attendance"].sum().reset_index()
att["Year"] = att["Year"].astype(int)
plt.figure(figsize=(12, 7))
sns.barplot(x=att["Year"], y=att["Attendance"], linewidth=1, edgecolor="k")
plt.grid(True)
plt.title("Attendance by Year", color='b')
plt.show()


# # Average attendence by year
# * A total of 3.43 million people watched the 64 games of the 2014 FIFA World Cup in Brazil live in the stadium. This meant that the average attendance per game was 53,758 , the highest average since the 1994 World Cup in the United States where average attendance is 68,991 per game. 

# In[131]:


att1 = matches.groupby("Year")["Attendance"].mean().reset_index()
att1["Year"] = att1["Year"].astype(int)
plt.figure(figsize=(12,7))
ax = sns.pointplot(att1["Year"],att1["Attendance"],color="w")
ax.set_facecolor("k")
plt.grid(True,color="grey",alpha=.3)
plt.title("Average attendence by year",color='b')
plt.show()


# # Total goals scored by year

# In[132]:


plt.figure(figsize=(13,7))
cups["Year1"] = cups["Year"].astype(str)
ax = plt.scatter("Year1","GoalsScored",data=cups,
            c=cups["GoalsScored"],cmap="inferno",
            s=900,alpha=.7,
            linewidth=2,edgecolor="k",)
plt.xticks(cups["Year1"].unique())
plt.yticks(np.arange(60,200,20))
plt.title('Total goals scored by year',color='b')
plt.show()


# # Total matches played and qualified teams  by year
# * In the tournaments between 1934 and 1978, 16 teams competed in each tournament, except in 1938, when Austria was absorbed into Germany after qualifying, leaving the tournament with 15 teams, and in 1950, when India, Scotland, and Turkey withdrew, leaving the tournament with 13 teams
# * The tournament was expanded to 24 teams in 1982, and then to 32 in 1998,also allowing more teams from Africa, Asia and North America to take part. 

# In[133]:


plt.figure(figsize=(12, 7))

sns.barplot(x=cups["Year"], y=cups["MatchesPlayed"], 
            linewidth=1, edgecolor="k", color="b", label="Total matches played")

sns.barplot(x=cups["Year"], y=cups["QualifiedTeams"], 
            linewidth=1, edgecolor="k", color="r", label="Total qualified teams", alpha=0.7)
plt.legend(loc="best", prop={"size": 13})
plt.title("Qualified teams by year", color='b')
plt.grid(True)

plt.show()


# # Matches with highest number of attendance

# In[134]:



h_att = matches.sort_values(by="Attendance", ascending=False)[:10]

h_att = h_att[['Year', 'Datetime', 'Stadium', 'City', 'Home Team Name',
               'Home Team Goals', 'Away Team Goals', 'Away Team Name', 
               'Attendance', 'MatchID']]
h_att["Datetime"] = h_att["Datetime"].str.split("-").str[0]

h_att["mt"] = h_att["Home Team Name"] + " vs " + h_att["Away Team Name"]
plt.figure(figsize=(10, 9))
ax = sns.barplot(y=h_att["mt"], x=h_att["Attendance"], palette="gist_ncar", 
                 linewidth=1, edgecolor="k")
plt.ylabel("Teams")
plt.xlabel("Attendance")
plt.title("Matches with Highest Number of Attendance", color='b')
plt.grid(True)
for i, row in h_att.iterrows():
    ax.text(row["Attendance"] * 0.7, i, f"Stadium: {row['Stadium']}, Date: {row['Datetime']}", 
            fontsize=12, color="white", weight="bold")
plt.show()


# # Stadiums with highest average attendance

# In[135]:


matches["Year"] = matches["Year"].astype(int)
matches["Datetime"] = matches["Datetime"].str.split("-").str[0]

matches["Stadium"] = matches["Stadium"].str.replace('Estadio do Maracana', "Maracanã Stadium")
matches["Stadium"] = matches["Stadium"].str.replace('Maracan� - Est�dio Jornalista M�rio Filho', "Maracanã Stadium")
std = matches.groupby(["Stadium", "City"])["Attendance"].mean().reset_index().sort_values(by="Attendance", ascending=False)
plt.figure(figsize=(8, 9))
ax = sns.barplot(y=std["Stadium"][:14], x=std["Attendance"][:14], palette="cool", linewidth=1, edgecolor="k")
plt.grid(True)
for i, row in std[:14].iterrows():
    ax.text(row["Attendance"] * 0.7, i, f"City: {row['City']}", fontsize=14)
plt.title("Stadiums with Highest Average Attendance", color='b')

plt.show()


# # Cities that hosted highest world cup matches

# In[136]:


mat_c = matches["City"].value_counts().reset_index()
mat_c.columns = ['City', 'Matches']

plt.figure(figsize=(10, 8))
ax = sns.barplot(y=mat_c["City"][:15], x=mat_c["Matches"][:15], palette="plasma", linewidth=1, edgecolor="k")

plt.xlabel("Number of Matches")
plt.ylabel("City")
plt.grid(True)
plt.title("Cities with Maximum World Cup Matches", color='b')
for i, row in mat_c[:15].iterrows():
    ax.text(row["Matches"] * 0.7, i, f"Matches: {row['Matches']}", fontsize=13, color="w")

plt.show()


# # Average attendance by city

# In[137]:


ct_at = matches.groupby("City")["Attendance"].mean().reset_index()
ct_at = ct_at.sort_values(by="Attendance", ascending=False)

plt.figure(figsize=(10, 10))
ax = sns.barplot(x="Attendance", y="City", data=ct_at[:20], linewidth=1, edgecolor="k", palette="Spectral_r")
for i, row in ct_at[:20].iterrows():
    ax.text(row["Attendance"] * 0.7, i, f"Avg attendance: {np.around(row['Attendance'], 0)}", fontsize=12)
plt.grid(True)
plt.title("Average Attendance by City", color='b')

plt.show()


# # Teams with the most world cup final victories
# * The 20 World Cup tournaments have been won by eight national teams. Brazil have won five times, and they are the only team to have played in every tournament. The other World Cup winners are Germany and Italy, with four titles each; Argentina and inaugural winner Uruguay, with two titles each; and England, France, and Spain, with one title each.

# In[138]:


cups["Winner"] = cups["Winner"].replace("Germany FR", "Germany")
cups["Runners-Up"] = cups["Runners-Up"].replace("Germany FR", "Germany")
cou = cups["Winner"].value_counts().reset_index()
cou.columns = ['Country', 'Wins']
plt.figure(figsize=(12, 7))
sns.barplot(x="Country", y="Wins", data=cou, palette="jet_r", linewidth=2, edgecolor="k")
plt.grid(True)
plt.ylabel("Number of Times")
plt.xlabel("Country")
plt.title("Teams with the Most World Cup Victories", color='b')
plt.xticks(color="navy", fontsize=12)
plt.show()


# # World cup final results by nation

# In[139]:


cou_w = cou.copy()
cou_w.columns = ["country", "count"]
cou_w["type"] = "WINNER"
cou_r = cups["Runners-Up"].value_counts().reset_index()
cou_r.columns = ["country", "count"]
cou_r["type"] = "RUNNER - Up"
cou_t = pd.concat([cou_w, cou_r], axis=0)
plt.figure(figsize=(8, 10))
sns.barplot(x="count", y="country", data=cou_t, hue="type", palette=["lime", "r"], linewidth=1, edgecolor="k")
plt.grid(True)
plt.legend(loc="center right", prop={"size": 14})
plt.title("Final Results by Nation", color='b')
plt.show()


# # World cup final result for third and fourth place by nation

# In[140]:


thrd = cups["Third"].value_counts().reset_index()
thrd.columns = ["team", "count"]
thrd["type"] = "THIRD PLACE"
frth = cups["Fourth"].value_counts().reset_index()
frth.columns = ["team", "count"]
frth["type"] = "FOURTH PLACE"
plcs = pd.concat([thrd, frth], axis=0)

plt.figure(figsize=(10, 10))
sns.barplot(x="count", y="team", data=plcs, hue="type", linewidth=1, edgecolor="k", palette=["grey", "r"])
plt.grid(True)
plt.xticks(np.arange(0, plcs["count"].max() + 1, 1))
plt.title("World Cup Final Results for Third and Fourth Place by Nation", color='b')
plt.legend(loc="center right", prop={"size": 12})

plt.show()


# # Teams with the most world cup matches

# In[141]:


matches["Home Team Name"] = matches["Home Team Name"].str.replace("Germany FR", "Germany")
matches["Away Team Name"] = matches["Away Team Name"].str.replace("Germany FR", "Germany")

ht = matches["Home Team Name"].value_counts().reset_index()
ht.columns = ["team", "matches"]

at = matches["Away Team Name"].value_counts().reset_index()
at.columns = ["team", "matches"]

mt = pd.concat([ht, at], axis=0)
mt = mt.groupby("team")["matches"].sum().reset_index().sort_values(by="matches", ascending=False)

plt.figure(figsize=(10, 13))
ax = sns.barplot(x="matches", y="team", data=mt[:25], palette="gnuplot_r", linewidth=1, edgecolor="k")
plt.grid(True)
plt.title("Teams with the Most Matches", color='b')

for i, count in enumerate(mt["matches"][:25]):
    ax.text(count * 0.7, i, f"Matches played: {count}", fontsize=12, color="black")

plt.show()


# # Teams with the most tournament participations

# In[142]:


hy = matches[["Year", "Home Team Name"]]
hy.columns = ["year", "team"]
hy["type"] = "HOME TEAM"
ay = matches[["Year", "Away Team Name"]]
ay.columns = ["year", "team"]
ay["type"] = "AWAY TEAM"

home_away = pd.concat([hy, ay], axis=0)
yt = home_away.groupby(["year", "team"]).count().reset_index()
yt = yt["team"].value_counts().reset_index()
yt.columns = ["team", "count"]
plt.figure(figsize=(10, 8))
ax = sns.barplot(x="count", y="team", data=yt[:15], linewidth=1, edgecolor="k")
for i, count in enumerate(yt["count"][:15]):
    ax.text(count * 0.7, i, f"Participated: {count} times", fontsize=14, color="k")
plt.grid(True)
plt.title("Teams with the Most Tournament Participations", color='b')

plt.show()


# # Distribution of home and away goals

# In[143]:


plt.figure(figsize=(12,13))
plt.subplot(211)
sns.distplot(matches["Home Team Goals"],color="b",rug=True)
plt.xticks(np.arange(0,12,1))
plt.title("Distribution of Home Team Goals",color='b')


plt.subplot(212)
sns.distplot(matches["Away Team Goals"],color="r",rug=True)
plt.xticks(np.arange(0,9,1))
plt.title("Distribution of Away Team Goals",color='b')
plt.show()


# # Distribution of Half time Home and Away Team Goals

# In[144]:


plt.figure(figsize=(12,15))
matches = matches.rename(columns={'Half-time Home Goals':"first half home goals",
                                  'Half-time Away Goals':"first half away goals"})

matches["second half home goals"] = matches["Home Team Goals"] - matches["first half home goals"]
matches["second half away goals"] = matches["Away Team Goals"] - matches["first half away goals"]

plt.subplot(211)
sns.kdeplot(matches["first half home goals"],color="b",linewidth=2)
sns.kdeplot(matches["second half home goals"],color="r",linewidth=2)
plt.xticks(np.arange(0,9,1))
plt.title("Distribution of first and second Half - Home Team Goals",color='b')

plt.subplot(212)
sns.kdeplot(matches["first half away goals"],color="b",linewidth=2)
sns.kdeplot(matches["second half away goals"],color="r",linewidth=2)
plt.xticks(np.arange(0,9,1))
plt.title("Distribution of first and second Half - Away Team Goals",color='b')
plt.show()


# # Home and away goals by year

# In[145]:


gh = matches[["Year","Home Team Goals"]]
gh.columns = ["year","goals"]
gh["type"] = "Home Team Goals"

ga = matches[["Year","Away Team Goals"]]
ga.columns = ["year","goals"]
ga["type"] = "Away Team Goals"

gls = pd.concat([ga,gh],axis=0)

plt.figure(figsize=(13,8))
sns.violinplot(gls["year"],gls["goals"],
               hue=gls["type"],split=True,inner="quart",palette="husl")
plt.grid(True)
plt.title("Home and away goals by year",color='b')
plt.show()


# # First half home and away goals by year

# In[146]:


hhg = matches[["Year",'first half home goals']]
hhg.columns = ["year","goals"]
hhg["type"] = 'first half home goals'

hag = matches[["Year",'first half away goals']]
hag.columns = ["year","goals"]
hag["type"] = 'first half away goals'

h_time = pd.concat([hhg,hag],axis=0)

plt.figure(figsize=(13,8))
sns.violinplot(h_time["year"],h_time["goals"],hue=h_time["type"],
               split=True,inner="quart",palette="gist_ncar")
plt.grid(True)
plt.title(" first half  home and away goals by year",color='b')
plt.show()


# # second half home and away goals by year

# In[147]:


shg = matches[["Year",'second half home goals']]
shg.columns = ["year","goals"]
shg["type"] = 'second half home goals'

sag = matches[["Year",'second half away goals']]
sag.columns = ["year","goals"]
sag["type"] = 'second half away goals'

s_time = pd.concat([shg,sag],axis=0)

plt.figure(figsize=(13,8))
sns.violinplot(s_time["year"],s_time["goals"],hue=s_time["type"],
               split=True,inner="quart",palette="gist_ncar")
plt.title("second half home and away goals by year",color='b')
plt.grid(True)
plt.show()


# # Match outcomes by home and away teams

# In[148]:


print(matches.columns)
matches[['Home Team Name', 'Home Team Goals', 'Away Team Goals', 'Away Team Name']]
def win_label(row):
    if row["Home Team Goals"] > row["Away Team Goals"]:
        return row["Home Team Name"]
    elif row["Home Team Goals"] < row["Away Team Goals"]:
        return row["Away Team Name"]
    else:
        return "DRAW"
def lst_label(row):
    if row["Home Team Goals"] < row["Away Team Goals"]:
        return row["Home Team Name"]
    elif row["Home Team Goals"] > row["Away Team Goals"]:
        return row["Away Team Name"]
    else:
        return "DRAW"
matches["win_team"] = matches.apply(win_label, axis=1)
matches["lost_team"] = matches.apply(lst_label, axis=1)
if 'outcome' not in matches.columns:
    matches['outcome'] = matches.apply(lambda row: "DRAW" if row["Home Team Goals"] == row["Away Team Goals"] else ("Home Win" if row["Home Team Goals"] > row["Away Team Goals"] else "Away Win"), axis=1)

win_counts = matches["win_team"].value_counts().reset_index()
lost_counts = matches["lost_team"].value_counts().reset_index()
wl = win_counts.merge(lost_counts, left_on="index", right_on="index", how="left")
wl = wl[wl["index"] != "DRAW"]
wl.columns = ["team", "wins", "loses"]

wl = wl.reset_index(drop=True)
print(wl)


# In[149]:


cols = ['wins', 'loses', 'draws']
length = len(cols)

plt.figure(figsize=(8, 18))

for i, j in itertools.zip_longest(cols, range(length)):
    plt.subplot(3, 1, j + 1)

    edge_colors = ['k'] * 10 
    
    ax = sns.barplot(x=i, y="team", data=wl1.sort_values(by=i, ascending=False)[:10],
                     linewidth=1, edgecolor=edge_colors, palette="husl")
    
    for k, l in enumerate(wl1.sort_values(by=i, ascending=False)[:10][i]):
        ax.text(.7, k, l, fontsize=13)
    
    plt.grid(True)
    plt.title("Countries with maximum " + i, color='b')

plt.tight_layout()
plt.show()


# # Teams with highest fifa world cup goals

# In[150]:


tt_gl_h = matches.groupby("Home Team Name")["Home Team Goals"].sum().reset_index()
tt_gl_h.columns = ["team", "goals"]
tt_gl_a = matches.groupby("Away Team Name")["Away Team Goals"].sum().reset_index()
tt_gl_a.columns = ["team", "goals"]
total_goals = pd.concat([tt_gl_h, tt_gl_a], axis=0)
total_goals = total_goals.groupby("team")["goals"].sum().reset_index()
total_goals = total_goals.sort_values(by="goals", ascending=False)
total_goals["goals"] = total_goals["goals"].astype(int)

plt.figure(figsize=(10, 12))

edge_colors = ['k'] * 20 

ax = sns.barplot(x="goals", y="team", data=total_goals[:20], palette="cool", 
                 linewidth=1, edgecolor=edge_colors)
for i, j in enumerate("SCORED  " + total_goals["goals"][:20].astype(str) + "  GOALS"):
    ax.text(.7, i, j, fontsize=10, color="k")

plt.title("Teams with Highest FIFA World Cup Goals", color='b')
plt.grid(True)
plt.show()


# # Total goals scored during games by year

# In[151]:


matches["total_goals"] = matches["Home Team Goals"] + matches["Away Team Goals"]

plt.figure(figsize=(13,8))
sns.boxplot(y=matches["total_goals"],x=matches["Year"])
plt.grid(True)
plt.title("Total goals scored during game by year",color='b')
plt.show()


# # Team comparator

# In[152]:


matches_played = mt.copy()
mat_new = matches_played.merge(lst, left_on="team", right_on="index", how="left")
mat_new = mat_new.merge(win, left_on="team", right_on="index", how="left")
mat_new = mat_new[["team", "matches", "lost_team", "win_team"]]
mat_new = mat_new.fillna(0)
mat_new["win_team"] = mat_new["win_team"].astype(int)
mat_new["draws"] = mat_new["matches"] - (mat_new["lost_team"] + mat_new["win_team"])
mat_new = mat_new.merge(total_goals, left_on="team", right_on="team", how="left")
mat_new = mat_new.rename(columns={"win_team": "wins", "lost_team": "loses"})

def team_compare(team1, team2):
    lst = [team1, team2]
    dat = mat_new[mat_new["team"].isin(lst)]
    plt.figure(figsize=(12, 8))
    cols = ["matches", "goals", "wins", "loses", "draws"]  
    length = len(cols)

    for j, col in enumerate(cols):
        ax = plt.subplot(length, 1, j + 1)  
        sns.barplot(x=dat[col], y=dat["team"], palette=["royalblue", "red"],
                    linewidth=2, edgecolor="k")

        plt.ylabel("")
        plt.yticks(fontsize=13)
        plt.grid(True, color="grey", alpha=.3)
        plt.title(col.capitalize(), color="b", fontsize=15)
        plt.subplots_adjust(wspace=.3, hspace=.5)

        for k, val in enumerate(dat[col].values):
            ax.text(val + 0.05, k, str(int(val)), weight="bold", fontsize=12, color="black")  # Adjusted text position

    plt.show()


# # Portugal & Argentina

# In[153]:


team_compare("Portugal","Argentina")


# # Spain & Italy

# In[154]:


team_compare("Italy","Spain")


# # Brazil & Germany

# In[155]:


team_compare("Brazil","Germany")


# # Referee's with most matches

# In[156]:


ref = matches["Referee"].value_counts().reset_index()
ref = ref.sort_values(by="Referee",ascending=False)

plt.figure(figsize=(10,10))
sns.barplot("Referee","index",data=ref[:20],linewidth=1,edgecolor="k")
plt.xlabel("count")
plt.ylabel("Refree name")
plt.grid(True)
plt.title("Referee's with most matches",color='b')
plt.show()


# # Goals per game by top countries 

# In[157]:


mat_new["goals_per_match"] = mat_new["goals"] / mat_new["matches"]
cou_lst = mat_new.sort_values(by="wins",ascending=False)[:15]["team"].tolist()
cou_gpm = mat_new[mat_new["team"].isin(cou_lst)]
cou_gpm = cou_gpm.sort_values(by="goals_per_match",ascending=False)

plt.figure(figsize=(10,8))
ax = sns.barplot("goals_per_match","team",
                 linewidth=1,
                 edgecolor=["k"]*len(cou_gpm),
                 data=cou_gpm,
                 palette="Spectral")

for i,j in enumerate(np.round(cou_gpm["goals_per_match"],2).astype(str) + "  Goals per game"):
    ax.text(.1,i,j,color="k",weight = "bold")
    
plt.grid(True)
plt.title("Goals per match for countries with highest wins",color='b')
plt.show()


# # Interactions between teams

# In[158]:


import networkx as nx 

def interactions(year, color):
    df = matches[matches["Year"] == year][["Home Team Name", "Away Team Name"]]
    G = nx.from_pandas_edgelist(df, "Home Team Name", "Away Team Name")
    plt.figure(figsize=(10, 9))
    nx.draw_kamada_kawai(G,
                          with_labels=True,
                          node_size=2500,
                          node_color=color,
                          node_shape="h",
                          edge_color="k",
                          linewidths=2,
                          font_size=10,
                          alpha=0.8)
    plt.title(f"Interaction Between Teams: {year}", fontsize=13, color="b")
    plt.axis('off')  
    plt.show()  


# # Interactions between teams for year 2014

# In[159]:


interactions(2014,"r")


# # 1994

# In[160]:


interactions(1994,"royalblue")


# # Interactions between teams for year 1950

# In[161]:


interactions(1950,"lawngreen")


# # Interactions between teams for year 1930

# In[162]:


interactions(1930,"Orange")


# # Total world cup matches played in each country

# In[163]:


ysc = matches[["Year", "Stadium", "City", "MatchID"]]
cy = cups[["Year", "Country"]]
ysc = ysc.merge(cy, on="Year", how="left")
ysc["std_cty"] = ysc["Stadium"] + " , " + ysc["City"]
cnt_mat = ysc.groupby("Country")["MatchID"].nunique().reset_index()
cnt_mat = cnt_mat.sort_values(by="MatchID", ascending=False)
plt.figure(figsize=(10, 8))
ax = sns.barplot(x="MatchID", y="Country", data=cnt_mat, linewidth=1, edgecolor="k")
for i, count in enumerate(cnt_mat["MatchID"]):
    ax.text(count + 0.1, i, str(count), color='k', fontsize=12)
plt.title("Total World Cup Matches Played in Each Country", color='b')
plt.grid(True)
plt.xlabel("Number of Matches")
plt.ylabel("Country")
plt.show()


# # Stadiums by countries

# In[164]:


ysc["Country_yr"] = ysc["Country"] + " - " + ysc["Year"].astype(str)

def stadium_country(country, color):
    dat2 = ysc[ysc["Country"] == country]
    if dat2.empty:
        print(f"No data available for {country}.")
        return
    
    plt.figure(figsize=(10, 8))
    H = nx.from_pandas_edgelist(dat2, "Country", "Stadium")
    nx.draw_kamada_kawai(H,
                          with_labels=True,
                          node_size=2500,
                          node_color=color,
                          node_shape="s",
                          edge_color="k",
                          linewidths=7,
                          font_size=13,
                          alpha=0.8)

    plt.title(f"Stadiums in {country}", fontsize=16, color='b') 
    plt.show()


# # Stadiums - Germany

# In[165]:


stadium_country("Germany","c")


# # Stadiums - Brazil

# In[166]:


stadium_country("Brazil","b")


# # Stadiums - Mexico

# In[167]:


stadium_country("Mexico","r")


# # Stadiums - USA

# In[168]:


stadium_country("USA","grey")


# In[ ]:





# In[ ]:




