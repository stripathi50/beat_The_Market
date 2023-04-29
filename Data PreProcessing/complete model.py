import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fights_df = pd.read_csv('masterdataframe.csv', parse_dates=True)
fighters_df= pd.read_csv('pro_mma_fighters.csv')

ufc_fighters = pd.DataFrame(fights_df.drop_duplicates("fighter")["fighter"])
print(f"From {fights_df.date.min()} to {fights_df.date.max()} there were {len(fights_df)//2} fights in total, included {ufc_fighters.shape[0]} fighters")
fights_df = fights_df.loc[:,:"ground_strikes_def_differential"]
fights_df["year"] = pd.DatetimeIndex(fights_df['date']).year

fights_by_year = pd.DataFrame(fights_df.groupby("year")["result"].count() // 2).rename(
    columns={"result": "no of fights"})
values = fights_by_year["no of fights"]
colors = ['navy' if (y < max(values)) else 'green' for y in values]
sns.set(font_scale=1.2)
plt.figure(figsize=(18, 10))
bar = sns.barplot(x=fights_by_year.index, y=values, palette=colors)

ax = plt.gca()
y_max = values.max()
ax.set_ylim(1)
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2., p.get_height(), p.get_height(),
            fontsize=10, color='black', ha='center', va='bottom')

plt.xlabel('Year')
plt.ylabel('no of fights')
plt.title('Fights Per Year', weight='bold')
plt.show()

replace_map = {
    "M-DEC": "DEC",
    "S-DEC" : "DEC",
    "U-DEC": "DEC",
    "DRAW": "DEC"

}
df1=fights_df.copy()
df1["method"] = df1["method"].replace(replace_map)

df1 = df1.groupby("year")["method"].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

ufc_fighters = ufc_fighters.merge(fighters_df[["fighter_name", "country"]], left_on="fighter", right_on= "fighter_name", how='left').drop(columns = ['fighter_name'])
print(fights_df.division.value_counts(normalize = True))

fights_df["gender"] = fights_df.division.apply(lambda x: "female" if x[0:3]=="Wom" else "male")

fights_df = fights_df[fights_df.gender!="female"]
fights_df = fights_df[~fights_df["division"].isin(["Open Weight", "Catch Weight", "Super Heavyweight"])]
weights=["Flyweight", "Bantamweight", "Featherweight", "Lightweight", "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight"]


import numpy as np
pivot = fights_df.pivot_table(values="result", index="division", columns="method", aggfunc="count", fill_value = 0).apply(lambda x: x//2)
pivot["total_fights"] = pivot.iloc[:,0:7].sum(axis=1)
pivot

time_by_division = pd.DataFrame(fights_df.groupby("division")["total_comp_time"].sum()//2)

pivot = pivot.join(time_by_division)
pivot["sub/time"] = pivot["total_comp_time"]//pivot["SUB"]
pivot["KO/TKO_time"] = pivot["total_comp_time"]//pivot["KO/TKO"]

fights_df["opponent_control"] = fights_df["control"]/fights_df["control_differential"]
fights_df["total_control"] = fights_df["control"] + fights_df["opponent_control"]
fights_df["clear_time_on_feet"] = fights_df["total_comp_time"] - fights_df["total_control"]
fights_df["strikes_per_min_standup"] =(fights_df["total_strikes_attempts"]/fights_df["clear_time_on_feet"])*60

gr_by_fights=fights_df.groupby("fight_url")[["control","total_comp_time", "division"]].agg({"control": "sum", "total_comp_time":"sum", "division": "first"})
gr_by_fights["total_comp_time"] = gr_by_fights["total_comp_time"].apply(lambda x: x//2)
gr_by_fights["standup_time"] = gr_by_fights["total_comp_time"]-gr_by_fights["control"]

gr_by_fights = gr_by_fights[["division", "standup_time", "control"]]
gr_by_fights_melted = pd.melt(gr_by_fights, id_vars=["division"], value_vars =['standup_time', 'control'])

fights_df["opponent_sig_strikes"] = fights_df["sig_strikes_landed"]/fights_df["sig_strikes_landed_differential"]
fights_df["control_diff"] = fights_df["control"]  - fights_df["opponent_control"]
fights_df["sig_strike_diff_s"] = fights_df["sig_strikes_landed"]  - fights_df["opponent_sig_strikes"]

ufc_fighters = ufc_fighters.drop("country", axis=1)
ufc_fighters["no_of_matches"] = ufc_fighters.fighter.map(fights_df.fighter.value_counts())
ufc_fighters=ufc_fighters.drop_duplicates("fighter")
ufc_fighters = ufc_fighters.set_index("fighter")
fighters_to_analyze = pd.DataFrame(ufc_fighters.index[ufc_fighters["no_of_matches"]>3])


def skills(fighter_name):
    time_total = fights_df["total_comp_time"][(fights_df["fighter"] == fighter_name)].sum()
    time_ground_control = fights_df["control"][(fights_df["fighter"] == fighter_name)].sum()
    no_of_fights = ufc_fighters.loc[fighter_name]["no_of_matches"]
    time_standup = time_total - fights_df["control"][(fights_df["fighter"] == fighter_name)].sum()

    def ground_skills_def():
        sub_losses = fights_df["result"][(fights_df["fighter"] == fighter_name) & (fights_df["result"] == 0) & (
                    fights_df["method"] == "SUB")].count()
        sub_losses_ratio = sub_losses / no_of_fights

        ground_op_control = fights_df["opponent_control"][(fights_df["fighter"] == fighter_name)].sum()
        op_control_ratio = ground_op_control / time_total

        g_a_p_deff = g_a_p = fights_df["ground_strikes_def"][(fights_df["fighter"] == fighter_name)].mean() / 100
        takedown_def = fights_df["takedowns_def"][fights_df["fighter"] == fighter_name].mean() / 100
        reversal_skill = (fights_df["reversals"][fights_df["fighter"] == fighter_name].sum() / no_of_fights)
        skill_total = (takedown_def + reversal_skill - (sub_losses_ratio * 2) + g_a_p_deff) * (1 - op_control_ratio)
        return skill_total if skill_total > 0 else 0

    def ground_skills_att():
        sub_winner = fights_df["result"][(fights_df["fighter"] == fighter_name) & (fights_df["result"] == 1) & (
                    fights_df["method"] == "SUB")].count()
        sub_winner_ratio = sub_winner / no_of_fights
        ground_control_coef = (1 - (time_ground_control / time_total))

        g_a_p = fights_df["ground_strikes_landed"][(fights_df["fighter"] == fighter_name)].sum() / time_total
        takedown_att = fights_df[fights_df["fighter"] == fighter_name]["takedowns_accuracy"].mean() / 100

        skill_total = (takedown_att + g_a_p + (sub_winner_ratio * 2)) / ground_control_coef
        return (skill_total)

    def standing_skills_att():
        ko_wins = fights_df["result"][(fights_df["fighter"] == fighter_name) & (fights_df["result"] == 1) & (
                    fights_df["method"] == "KO/TKO")].count()
        ko_wins_ratio = ko_wins / no_of_fights
        standup_ratio = time_standup / time_total
        ground_strikes = fights_df["ground_strikes_attempts"][fights_df["fighter"] == fighter_name].sum()

        sig_strikes_eff = (fights_df["sig_strikes_landed"][
                               (fights_df["fighter"] == fighter_name)].sum() - ground_strikes) / time_standup
        hand_speed = (fights_df[fights_df["fighter"] == fighter_name][
                          "total_strikes_attempts"].sum() - ground_strikes) / time_standup
        clinch = fights_df[fights_df["fighter"] == fighter_name]["clinch_strikes_landed"].sum() / time_standup / \
                 fights_df[fights_df["fighter"] == fighter_name]["clinch_strikes_accuracy"].mean() / 100
        accuracy = (fights_df[fights_df["fighter"] == fighter_name]["total_strikes_accuracy"].mean() +
                    fights_df[fights_df["fighter"] == fighter_name]["distance_strikes_accuracy"].mean()) / 200
        total_skill = (clinch + (ko_wins_ratio * 2) + (sig_strikes_eff + hand_speed + accuracy) / 2)
        return (total_skill)

    def standing_skills_def():
        ko_losses = fights_df["result"][(fights_df["fighter"] == fighter_name) & (fights_df["result"] == 0) & (
                    fights_df["method"] == "KO/TKO")].count()
        ko_losses_ratio = ko_losses / no_of_fights

        standup_ratio = time_standup / time_total

        sig_strikes_def = fights_df["sig_strikes_def"][(fights_df["fighter"] == fighter_name)].mean() / 100
        clinch_strikes_def = fights_df[fights_df["fighter"] == fighter_name]["clinch_strikes_def"].mean() / 100
        distance_strikes_def = fights_df[fights_df["fighter"] == fighter_name]["distance_strikes_def"].mean() / 100
        total_skill = (
                                  clinch_strikes_def - ko_losses_ratio * 2 + sig_strikes_def + distance_strikes_def) * standup_ratio

        return total_skill if total_skill > 0 else 0

    def stamina():
        wins_second_round = fights_df["result"][
            (fights_df["fighter"] == fighter_name) & (fights_df["total_comp_time"] > 300) & (
                        fights_df["total_comp_time"] <= 600)].sum()
        wins_third_round = fights_df["result"][
            (fights_df["fighter"] == fighter_name) & (fights_df["total_comp_time"] > 600) & (
                        fights_df["total_comp_time"] <= 900)].sum()
        wins_champ_round = fights_df["result"][
            (fights_df["fighter"] == fighter_name) & (fights_df["total_comp_time"] > 900)].sum()

        fights_second_round = fights_df["result"][
            (fights_df["fighter"] == fighter_name) & (fights_df["total_comp_time"] > 300) & (
                        fights_df["total_comp_time"] <= 600)].count()
        fights_third_round = fights_df["result"][
            (fights_df["fighter"] == fighter_name) & (fights_df["total_comp_time"] > 600) & (
                        fights_df["total_comp_time"] <= 900)].count()
        fights_champ_round = fights_df["result"][
            (fights_df["fighter"] == fighter_name) & (fights_df["total_comp_time"] > 900)].count()

        fights_second_round = fights_second_round if fights_second_round > 0 else 1
        fights_third_round = fights_third_round if fights_third_round > 0 else 1
        fights_champ_round = fights_champ_round if fights_champ_round > 0 else 1

        total_skill = np.array(
            [(0.2 * wins_second_round / fights_second_round), (0.6 * wins_third_round / fights_third_round),
             (wins_champ_round / fights_champ_round)])
        return total_skill.sum()

    return (ground_skills_def(), ground_skills_att(), standing_skills_def(), standing_skills_att(), stamina())


def form(fighter_name, datum):
  vysledek=''
  skore = 0
  koef =0.1
  result = ['W' if x==1 else 'L' for x in fights_df['result'][(fights_df['fighter']==fighter_name) & (fights_df['date']<datum)]]
  for vyhra in result[:-6:-1]:
    if vyhra =='W':
      skore+=koef
    else:
      skore-=koef
    koef+=0.1
    vysledek +=vyhra+' '
  vysledek=vysledek[:-1]
  return (vysledek,skore)

fighters_to_analyze.isna().sum()
fighters_to_analyze[["ground_def_skill","ground_att_skill", "stand_def_skill","stand_att_skill", "stamina"]] = fighters_to_analyze["fighter"].apply(skills).apply(pd.Series).astype(float)
fighters_to_analyze["stand_att_skill"] = fighters_to_analyze['stand_att_skill'].fillna(0)


fights_to_analyze = fights_df[["date", "fighter","opponent","result","method"]]

fights_to_analyze.loc[:,"form_skore_fighter"] = fights_to_analyze.apply(lambda x:form(x.fighter, x.date)[1], axis=1).astype(float)
fights_to_analyze.loc[:,"form_skore_opponent"] = fights_to_analyze.apply(lambda x:form(x.opponent, x.date)[1], axis=1).astype(float)

fights_to_analyze = fights_to_analyze.merge(fighters_to_analyze, on="fighter", how="inner")
fights_to_analyze = fights_to_analyze.merge(fighters_to_analyze,left_on="opponent",right_on="fighter", how="inner", suffixes=("_fighter","_opponent"))
fights_to_analyze=fights_to_analyze.drop("fighter_opponent",axis=1)
fights_to_analyze.head()


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X =fights_to_analyze[["ground_def_skill_fighter","ground_att_skill_fighter","stand_att_skill_fighter","stand_def_skill_fighter","stamina_fighter","form_skore_fighter",
                      "ground_def_skill_opponent", "ground_att_skill_opponent", "stand_att_skill_opponent","stand_def_skill_opponent","stamina_opponent", "form_skore_opponent" ]]
y = fights_to_analyze["result"]

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2,random_state=21)

reg = LogisticRegression()
reg.fit(xtrain, ytrain)

y_pred_lr = reg.predict(xtest)
log_train = round(reg.score(xtrain, ytrain) * 100, 2)
log_accuracy_MinMax = round(accuracy_score(y_pred_lr, ytest) * 100, 2)

print("Training Accuracy    :",log_train ,"%")
print("Model Accuracy Score :",log_accuracy_MinMax ,"%")


# cm = confusion_matrix(xtest, ytest)
# disp = ConfusionMatrixDisplay(cm,'Confusion Matrix')
# disp.plot()
# plt.title('Confusion Matrix');
# plt.show()


from sklearn.svm import SVC
svc = SVC(probability=True)
svc.fit(xtrain, ytrain)
y_pred_svc = svc.predict(xtest)

svc_train = round(svc.score(xtrain, ytrain) * 100, 2)
svc_accuracy = round(accuracy_score(y_pred_svc, ytest) * 100, 2)

print("Training Accuracy    :",svc_train ,"%")
print("Model Accuracy Score :",svc_accuracy ,"%")

# cm = confusion_matrix(xtest, ytest)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
# plt.title('Confusion Matrix');
# plt.show()


from sklearn import preprocessing
fighters_normalized = fighters_to_analyze.copy()
fighters_normalized.iloc[:,1:] =  preprocessing.normalize(fighters_normalized.iloc[:,1:])
fighters_normalized.iloc[:,1:]=fighters_normalized.iloc[:,1:].apply(lambda x:round(x*100,0))
fighters_normalized = fighters_normalized.set_index('fighter')
fighters_normalized.describe()

from math import pi


def proba(fighter, opponent):
    form_fighter = form(fighter, "2022-12-12")
    form_opponent = form(opponent, "2022-12-12")
    h1 = fighters_to_analyze[fighters_to_analyze.fighter == fighter].copy()
    h1.loc[:, 'form_skore'] = form_fighter[1]
    h2 = fighters_to_analyze[fighters_to_analyze.fighter == opponent].copy()
    h2.loc[:, 'form_skore'] = form_opponent[1]
    h1.loc[:, "opponent"] = opponent
    h1 = h1.merge(h2, left_on="opponent", right_on="fighter", how="inner", suffixes=("_fighter", "_opponent"))
    h1 = h1.loc[:,
         ["ground_def_skill_fighter", "ground_att_skill_fighter", "stand_att_skill_fighter", "stand_def_skill_fighter",
          "stamina_fighter", "form_skore_fighter",
          "ground_def_skill_opponent", "ground_att_skill_opponent", "stand_att_skill_opponent",
          "stand_def_skill_opponent", "stamina_opponent", "form_skore_opponent"]]

    probs = svc.predict_proba(h1)
    prob_fighter = probs[0][1]
    prob_opponent = probs[0][0]
    return prob_fighter, prob_opponent

def score(fighter):
  wins = fights_df['result'][(fights_df['fighter']==fighter)].sum()
  draws = fights_df['method'][(fights_df['fighter']==fighter) & (fights_df['method']=='DRAW')].count()
  lost = fights_df['result'][(fights_df['fighter']==fighter)].count() - wins - draws
  return f"W: {wins} - L: {lost} - D: {draws}"

def Head2Head(fighter, opponent):
    moje_prob = proba(fighter, opponent)
    form_fighter = form(fighter, "2022-12-12")
    form_opponent = form(opponent, "2022-12-12")
    prob_fighter = moje_prob[0]
    prob_opponent = moje_prob[1]
    x = 0
    y = 0
    fig = plt.figure(figsize=(19, 10))
    ax = fig.add_subplot(2, 3, 2)
    ax.pie(x=(prob_fighter, prob_opponent), labels=(fighter, opponent), colors=["#EE2C2C", "#6495ED"],
            autopct='%.00f%%',
            startangle=90,
            wedgeprops={'linewidth': 2, 'edgecolor': 'k'}, labeldistance=1.1,
            textprops={'fontsize': 14, 'weight': 'bold'})
    ax.add_artist(plt.Circle((0, 0), 0.35, fc='white', ec='black', lw=2))
    ax.annotate("% to win", xy=(x, y), va="center", ha="center")
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)

    print("UFC score Fighter: " + score(fighter))
    print("Last 5: " + form_fighter[0])
    print("calculated odds: " + str(round(1/prob_fighter,2)))

    print("UFC score Opponent: " + score(opponent))
    print("Last 5: " + form_opponent[0])
    print("calculated odds: " + str(round(1/prob_opponent,2)))



print(Head2Head('Conor McGregor','Khabib Nurmagomedov'))
print(Head2Head('Khamzat Chimaev','Nate Diaz'))
print(Head2Head('Israel Adesanya','Paulo Costa'))