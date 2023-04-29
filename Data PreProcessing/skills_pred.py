import pandas as pd
import numpy as np
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

df_mma2=pd.read_csv('/Users/sagartripathi/Documents/pythonProject/MMA_Betting/data/masterdataframe.csv')


df_mma2 = df_mma2.loc[:,:"ground_strikes_def_differential"]
df_mma2["year"] = pd.DatetimeIndex(df_mma2['date']).year
mma_fighters= pd.DataFrame(df_mma2.drop_duplicates("fighter")["fighter"])


mma_fighters = mma_fighters.merge(df_mma2[["fighter", "country"]], left_on="fighter", right_on= "fighter_name", how='left').drop(columns = ['fighter_name'])

# fighters_with_geo = mma_fighters.merge(coordinates, left_on='country', right_on='COUNTRY', how='left').drop(columns = ['ISO', 'COUNTRY','COUNTRYAFF', 'AFF_ISO'])
# fighters_with_geo=fighters_with_geo.dropna()


print(df_mma2.division.value_counts(normalize = True))

df_mma2["gender"] = df_mma2.division.apply(lambda x: "female" if x[0:3]=="Wom" else "male")

fights_df = df_mma2[df_mma2.gender!="female"]
fights_df = fights_df[~fights_df["division"].isin(["Open Weight", "Catch Weight", "Super Heavyweight"])]
weights=["Flyweight", "Bantamweight", "Featherweight", "Lightweight", "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight"]


pivot = df_mma2.pivot_table(values="result", index="division", columns="method", aggfunc="count", fill_value = 0).apply(lambda x: x//2)
pivot["total_fights"] = pivot.iloc[:,0:7].sum(axis=1)
pivot

time_by_division = pd.DataFrame(df_mma2.groupby("division")["total_comp_time"].sum()//2)

pivot = pivot.join(time_by_division)
pivot["sub/time"] = pivot["total_comp_time"]//pivot["SUB"]
pivot["KO/TKO_time"] = pivot["total_comp_time"]//pivot["KO/TKO"]




df_mma2["opponent_control"] = df_mma2["control"]/df_mma2["control_differential"]
df_mma2["total_control"] = df_mma2["control"] + df_mma2["opponent_control"]
df_mma2["clear_time_on_feet"] = df_mma2["total_comp_time"] - df_mma2["total_control"]
df_mma2["strikes_per_min_standup"] =(df_mma2["total_strikes_attempts"]/df_mma2["clear_time_on_feet"])*60
gr_by_fights=df_mma2.groupby("fight_url")[["control","total_comp_time", "division"]].agg({"control": "sum", "total_comp_time":"sum", "division": "first"})
gr_by_fights["total_comp_time"] = gr_by_fights["total_comp_time"].apply(lambda x: x//2)
gr_by_fights["standup_time"] = gr_by_fights["total_comp_time"]-gr_by_fights["control"]
gr_by_fights = gr_by_fights[["division", "standup_time", "control"]]
gr_by_fights_melted = pd.melt(gr_by_fights, id_vars=["division"], value_vars =['standup_time', 'control'])



df_mma2["opponent_sig_strikes"] = df_mma2["sig_strikes_landed"]/df_mma2["sig_strikes_landed_differential"]
df_mma2["control_diff"] = df_mma2["control"]  - df_mma2["opponent_control"]
df_mma2["sig_strike_diff_s"] = df_mma2["sig_strikes_landed"]  - df_mma2["opponent_sig_strikes"]

# mma_fighters = mma_fighters.drop("country", axis=1)
mma_fighters["no_of_matches"] = mma_fighters.fighter.map(mma_fighters.fighter.value_counts())
ufc_fighters=mma_fighters.drop_duplicates("fighter")
ufc_fighters = ufc_fighters.set_index("fighter")
fighters_to_analyze = pd.DataFrame(ufc_fighters.index[ufc_fighters["no_of_matches"]>3])


def skills(fighter_name):
    time_total = df_mma2["total_comp_time"][(df_mma2["fighter"] == fighter_name)].sum()
    time_ground_control = df_mma2["control"][(df_mma2["fighter"] == fighter_name)].sum()
    no_of_fights = mma_fighters.loc[fighter_name]["no_of_matches"]
    time_standup = time_total - df_mma2["control"][(df_mma2["fighter"] == fighter_name)].sum()

    def ground_skills_def():
        sub_losses = df_mma2["result"][(df_mma2["fighter"] == fighter_name) & (df_mma2["result"] == 0) & (
                    df_mma2["method"] == "SUB")].count()
        sub_losses_ratio = sub_losses / no_of_fights

        ground_op_control = df_mma2["opponent_control"][(df_mma2["fighter"] == fighter_name)].sum()
        op_control_ratio = ground_op_control / time_total

        g_a_p_deff = g_a_p = df_mma2["ground_strikes_def"][(df_mma2["fighter"] == fighter_name)].mean() / 100
        takedown_def = df_mma2["takedowns_def"][df_mma2["fighter"] == fighter_name].mean() / 100
        reversal_skill = (df_mma2["reversals"][df_mma2["fighter"] == fighter_name].sum() / no_of_fights)
        skill_total = (takedown_def + reversal_skill - (sub_losses_ratio * 2) + g_a_p_deff) * (1 - op_control_ratio)
        return skill_total if skill_total > 0 else 0

    def ground_skills_att():
        sub_winner = df_mma2["result"][(df_mma2["fighter"] == fighter_name) & (df_mma2["result"] == 1) & (
                    df_mma2["method"] == "SUB")].count()
        sub_winner_ratio = sub_winner / no_of_fights
        ground_control_coef = (1 - (time_ground_control / time_total))

        g_a_p = df_mma2["ground_strikes_landed"][(df_mma2["fighter"] == fighter_name)].sum() / time_total
        takedown_att = df_mma2[df_mma2["fighter"] == fighter_name]["takedowns_accuracy"].mean() / 100

        skill_total = (takedown_att + g_a_p + (sub_winner_ratio * 2)) / ground_control_coef
        return (skill_total)

    def standing_skills_att():
        ko_wins = df_mma2["result"][(df_mma2["fighter"] == fighter_name) & (df_mma2["result"] == 1) & (
                    df_mma2["method"] == "KO/TKO")].count()
        ko_wins_ratio = ko_wins / no_of_fights
        standup_ratio = time_standup / time_total
        ground_strikes = df_mma2["ground_strikes_attempts"][df_mma2["fighter"] == fighter_name].sum()

        sig_strikes_eff = (df_mma2["sig_strikes_landed"][
                               (df_mma2["fighter"] == fighter_name)].sum() - ground_strikes) / time_standup
        hand_speed = (df_mma2[df_mma2["fighter"] == fighter_name][
                          "total_strikes_attempts"].sum() - ground_strikes) / time_standup
        clinch = df_mma2[df_mma2["fighter"] == fighter_name]["clinch_strikes_landed"].sum() / time_standup / \
                 df_mma2[df_mma2["fighter"] == fighter_name]["clinch_strikes_accuracy"].mean() / 100
        accuracy = (df_mma2[df_mma2["fighter"] == fighter_name]["total_strikes_accuracy"].mean() +
                    df_mma2[df_mma2["fighter"] == fighter_name]["distance_strikes_accuracy"].mean()) / 200
        total_skill = (clinch + (ko_wins_ratio * 2) + (sig_strikes_eff + hand_speed + accuracy) / 2)
        return (total_skill)

    def standing_skills_def():
        ko_losses = df_mma2["result"][(df_mma2["fighter"] == fighter_name) & (df_mma2["result"] == 0) & (
                    df_mma2["method"] == "KO/TKO")].count()
        ko_losses_ratio = ko_losses / no_of_fights

        standup_ratio = time_standup / time_total

        sig_strikes_def = df_mma2["sig_strikes_def"][(df_mma2["fighter"] == fighter_name)].mean() / 100
        clinch_strikes_def = df_mma2[df_mma2["fighter"] == fighter_name]["clinch_strikes_def"].mean() / 100
        distance_strikes_def = df_mma2[df_mma2["fighter"] == fighter_name]["distance_strikes_def"].mean() / 100
        total_skill = (
                                  clinch_strikes_def - ko_losses_ratio * 2 + sig_strikes_def + distance_strikes_def) * standup_ratio

        return total_skill if total_skill > 0 else 0

    def stamina():
        wins_second_round = df_mma2["result"][
            (df_mma2["fighter"] == fighter_name) & (df_mma2["total_comp_time"] > 300) & (
                        df_mma2["total_comp_time"] <= 600)].sum()
        wins_third_round = df_mma2["result"][
            (df_mma2["fighter"] == fighter_name) & (df_mma2["total_comp_time"] > 600) & (
                        df_mma2["total_comp_time"] <= 900)].sum()
        wins_champ_round = df_mma2["result"][
            (df_mma2["fighter"] == fighter_name) & (df_mma2["total_comp_time"] > 900)].sum()

        fights_second_round = df_mma2["result"][
            (df_mma2["fighter"] == fighter_name) & (df_mma2["total_comp_time"] > 300) & (
                        df_mma2["total_comp_time"] <= 600)].count()
        fights_third_round = df_mma2["result"][
            (df_mma2["fighter"] == fighter_name) & (df_mma2["total_comp_time"] > 600) & (
                        df_mma2["total_comp_time"] <= 900)].count()
        fights_champ_round = df_mma2["result"][
            (df_mma2["fighter"] == fighter_name) & (df_mma2["total_comp_time"] > 900)].count()

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
  result = ['W' if x==1 else 'L' for x in df_mma2['result'][(df_mma2['fighter']==fighter_name) & (df_mma2['date']<datum)]]
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

fights_to_analyze = df_mma2[["date", "fighter","opponent","result","method"]]

fights_to_analyze.loc[:,"form_skore_fighter"] = fights_to_analyze.apply(lambda x:form(x.fighter, x.date)[1], axis=1).astype(float)
fights_to_analyze.loc[:,"form_skore_opponent"] = fights_to_analyze.apply(lambda x:form(x.opponent, x.date)[1], axis=1).astype(float)

fights_to_analyze = fights_to_analyze.merge(fighters_to_analyze, on="fighter", how="inner")
fights_to_analyze = fights_to_analyze.merge(fighters_to_analyze,left_on="opponent",right_on="fighter", how="inner", suffixes=("_fighter","_opponent"))
fights_to_analyze=fights_to_analyze.drop("fighter_opponent",axis=1)
fights_to_analyze.head()


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
# plot_confusion_matrix(reg, xtest, ytest);
plt.title('Confusion Matrix');

cm = confusion_matrix(reg, xtest, ytest)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()