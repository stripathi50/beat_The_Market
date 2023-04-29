#data modelling
#SciKit Learn Models

import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, r2_score
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv('updated_fighters3.csv')
data=data.drop(columns=['deviation', 'volatility'], axis=1)
data['gender'] = data['gender'].replace({'Male': 1, 'Female': 0})
data['division']=data['division'].replace({"Lightweight": 1, "Welterweight": 2,"Middleweight": 3,
                                           "Light Heavyweight": 4, "Heavyweight":5,"Featherweight":6,"Catch Weight":7,
                                           "Bantamweight":8,"Flyweight":9,"Women's Bantamweight":10,"Women's Strawweight":11
                                           ,"Women's Flyweight":12, "Women's Featherweight":13
                                           })
data['stance']=data['stance'].replace({'Southpaw': 1, 'Orthodox': 2,'Switch': 3, 'Open Stance': 4})
data['fighter_type']=data['fighter_type'].replace({'Volume_Striker': 1, 'Controller_Wrestler': 2,'Knockout_Artist': 3,
                                                   'Submission_Specialist': 4, 'Hybrid_Finisher': 5})

data['method']=data['method'].replace({'KO/TKO': 1, 'U-DEC': 2,'S-DEC': 3,
                                                   'SUB': 4, 'M-DEC': 5,'DRAW':6,'DQ':7})

data = data.drop('year', axis=1)

X=data[['division', 'stance',
       'method', 'total_comp_time', 'round', 'reach', 'height', 'age',
       'knockdowns_x', 'sub_attempts_x', 'reversals', 'control_x',
       'takedowns_attempts', 'total_strikes_attempts_x',
       'sub_attempts_per_min', 'takedowns_attempts_per_min',
       'sig_strikes_attempts_per_min', 'avg_knockdowns', 'avg_sub_attempts',
       'avg_reversals', 'avg_control', 'avg_takedowns_attempts',
       'avg_sig_strikes_landed', 'avg_sig_strikes_attempts',
       'avg_total_strikes_attempts', 'gender', 'fighter_type',
       'total_cumulative_wins', 'elo_rating', 'elo_win_probability', 'rating',
       'glicko2_win_probability']]

y = data['result']

corr_matrix = data.corr()

# Create a heatmap using seaborn
fig, ax = plt.subplots(figsize=(40,40))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax, annot_kws={"fontweight": "bold"})
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right',fontweight='bold', fontsize=20)
ax.set_title("Correlation Matrix Heatmap",fontweight='bold', fontsize=20)
plt.tight_layout()
plt.show()


corr_with_target = corr_matrix['result']
corr_with_target = corr_with_target[corr_with_target != 1] # Remove correlation with itself
corr_with_target = corr_with_target.abs().sort_values(ascending=False) # Sort by absolute value
print(corr_with_target)

# Standardize data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Compute correlation matrix
corr_matrix = pd.DataFrame(X_std).corr()

# Train random forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_std, y)

# Train Lasso model
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_std, y)

# Perform PCA
pca = PCA()
pca.fit(X_std)

# Get feature importances from different techniques
corr_importances = corr_matrix[0].abs().sort_values(ascending=False)
rf_importances = rf.feature_importances_
lasso_importances = lasso.coef_.tolist()
pca_importances = pca.explained_variance_ratio_.tolist()

# Create dataframe with feature importances
feature_importances = pd.DataFrame({'Features': X.columns,
                                    'Correlation': corr_importances,
                                    'Random Forest': rf_importances,
                                    'Lasso': lasso_importances,
                                    'PCA': pca_importances})

# Melt dataframe to plot bar graph
melted = pd.melt(feature_importances, id_vars=['Features'], var_name='Technique', value_name='Importance')

# Plot bar graph
fig, ax = plt.subplots(figsize=(40, 40))
sns.catplot(data=melted, x='Features', y='Importance', hue='Technique', kind='bar',legend_out=False)
plt.title("Feature Importance")
plt.xticks(rotation=90)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()


feature_importances = feature_importances.set_index('Features')
top_corr = feature_importances.nlargest(20, 'Correlation')
top_rf = feature_importances.nlargest(20, 'Random Forest')
top_lasso = feature_importances.nlargest(20, 'Lasso')
top_pca = feature_importances.nlargest(20, 'PCA')


#MODELLING IMPLEMENTATION ACCORDING TO IMPORTANT FEATURES SELECTED FROM ABOVE


def perceptron_prd(X_train,Y_train,X_test,Y_test,Models):
    perceptron = Perceptron(max_iter = 1000, tol = 0.001)
    perceptron.fit(X_train, Y_train)
    Y_pred = perceptron.predict(X_test)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    n = len(Y_test)
    p = len(top_rf.index)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    Models['Perceptron'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred),precision, recall,f1,r2,adj_r2]
    return

def ranforest_pred(X_train,Y_train,X_test,Y_test,Models):
    randomfor = RandomForestClassifier(max_features="sqrt",
                                       n_estimators = 700,
                                       max_depth = None,
                                       n_jobs=-1
                                      )
    randomfor.fit(X_train,Y_train)
    Y_pred = randomfor.predict(X_test)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    n = len(Y_test)
    p = len(top_rf.index)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    Models['Random Forests'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred),precision, recall,f1,r2,adj_r2]
    return

def decision_tree_pred(X_train,Y_train,X_test,Y_test,Models):
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)
    Y_pred = decision_tree.predict(X_test)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    n = len(Y_test)
    p = len(top_rf.index)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    Models['Decision Tree'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred),precision, recall,f1,r2,adj_r2]
    return

def SGDClass_pred(X_train,Y_train,X_test,Y_test,Models):
    sgd = SGDClassifier(max_iter = 1000, tol = 0.001)
    sgd.fit(X_train, Y_train)
    Y_pred = sgd.predict(X_test)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    n = len(Y_test)
    p = len(top_rf.index)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    Models['SGD Classifier'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred),precision, recall,f1,r2,adj_r2]
    return

def linSVC_pred(X_train,Y_train,X_test,Y_test,Models):
    linear_svc = LinearSVC()
    linear_svc.fit(X_train, Y_train)
    Y_pred = linear_svc.predict(X_test)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    n = len(Y_test)
    p = len(top_rf.index)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    Models['SVM'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred),precision, recall,f1,r2,adj_r2]
    return

def bayes_pred(X_train,Y_train,X_test,Y_test,Models):
    gaussian = GaussianNB()
    gaussian.fit(X_train, Y_train)
    Y_pred = gaussian.predict(X_test)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    n = len(Y_test)
    p = len(top_rf.index)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    Models['Bayes'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred),precision, recall,f1,r2,adj_r2]
    return

def KNearest(X_train,Y_train,X_test,Y_test,Models):
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    n = len(Y_test)
    p = len(top_rf.index)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    Models['KNN'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred),precision, recall,f1,r2,adj_r2]


def run_all_and_Plot_corr(df):
    Models_corr = dict()
    X_all = df[top_corr.index]
    y_all = df['result']
    X_train, X_test, Y_train, Y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=0)
    perceptron_prd(X_train,Y_train,X_test,Y_test,Models_corr)
    ranforest_pred(X_train,Y_train,X_test,Y_test,Models_corr)
    decision_tree_pred(X_train,Y_train,X_test,Y_test,Models_corr)
    SGDClass_pred(X_train,Y_train,X_test,Y_test,Models_corr)
    linSVC_pred(X_train,Y_train,X_test,Y_test,Models_corr)
    bayes_pred(X_train,Y_train,X_test,Y_test,Models_corr)
    KNearest(X_train,Y_train,X_test,Y_test,Models_corr)
    return Models_corr

def run_all_and_Plot_rf(df):
    Models_rf = dict()
    X_all = df[top_rf.index]
    y_all = df['result']
    X_train, X_test, Y_train, Y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=0)
    perceptron_prd(X_train,Y_train,X_test,Y_test,Models_rf)
    ranforest_pred(X_train,Y_train,X_test,Y_test,Models_rf)
    decision_tree_pred(X_train,Y_train,X_test,Y_test,Models_rf)
    SGDClass_pred(X_train,Y_train,X_test,Y_test,Models_rf)
    linSVC_pred(X_train,Y_train,X_test,Y_test,Models_rf)
    bayes_pred(X_train,Y_train,X_test,Y_test,Models_rf)
    KNearest(X_train,Y_train,X_test,Y_test,Models_rf)
    return Models_rf

def run_all_and_Plot_lasso(df):
    Models_ls = dict()
    X_all = df[top_lasso.index]
    y_all = df['result']
    X_train, X_test, Y_train, Y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=0)
    perceptron_prd(X_train,Y_train,X_test,Y_test,Models_ls)
    ranforest_pred(X_train,Y_train,X_test,Y_test,Models_ls)
    decision_tree_pred(X_train,Y_train,X_test,Y_test,Models_ls)
    SGDClass_pred(X_train,Y_train,X_test,Y_test,Models_ls)
    linSVC_pred(X_train,Y_train,X_test,Y_test,Models_ls)
    bayes_pred(X_train,Y_train,X_test,Y_test,Models_ls)
    KNearest(X_train,Y_train,X_test,Y_test,Models_ls)
    return Models_ls

def run_all_and_Plot_pca(df):
    Models = dict()
    X_all = df[top_pca.index]
    y_all = df['result']
    X_train, X_test, Y_train, Y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=0)
    perceptron_prd(X_train,Y_train,X_test,Y_test,Models)
    ranforest_pred(X_train,Y_train,X_test,Y_test,Models)
    decision_tree_pred(X_train,Y_train,X_test,Y_test,Models)
    SGDClass_pred(X_train,Y_train,X_test,Y_test,Models)
    linSVC_pred(X_train,Y_train,X_test,Y_test,Models)
    bayes_pred(X_train,Y_train,X_test,Y_test,Models)
    KNearest(X_train,Y_train,X_test,Y_test,Models)
    return Models


def plot_BarChart(dict,Feature_imp_method):
    labels = tuple(dict.keys())
    y_pos = np.arange(len(labels))
    values = [dict[n][0] for n in dict]
    plt.bar(y_pos, values, align='center', alpha=0.5)
    plt.xticks(y_pos, labels,rotation='vertical')
    plt.ylabel('accuracy')
    plt.title(f'Accuracy of different models {Feature_imp_method}')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(dict,Feature_imp_method):
    count = 1
    fig = plt.figure(figsize=(10,10))
    for model in dict:
        cm = dict[model][1]
        labels = ['W','L','N','D']
        ax = fig.add_subplot(4,4,count)
        cax = ax.matshow(cm)
        plt.title(model,y=-0.8)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        # plt.subplot(2,2,count)
        count+=1
    plt.title(f'Confusion matrix on selected {Feature_imp_method}',loc="center",pad=20)
    plt.tight_layout()
    plt.show()


print(run_all_and_Plot_pca(data))
print(run_all_and_Plot_rf(data))
print(run_all_and_Plot_lasso(data))
print(run_all_and_Plot_corr(data))

plot_BarChart(run_all_and_Plot_pca(data),'important features PCA')
plot_BarChart(run_all_and_Plot_rf(data),'important features Random Forest')
plot_BarChart(run_all_and_Plot_lasso(data),'important features Lasso')
plot_BarChart(run_all_and_Plot_corr(data),'important features Correlation')

plot_confusion_matrix(run_all_and_Plot_pca(data),'important features PCA')
plot_confusion_matrix(run_all_and_Plot_rf(data),'important features Random Forest')
plot_confusion_matrix(run_all_and_Plot_lasso(data),'important features Lasso')
plot_confusion_matrix(run_all_and_Plot_corr(data),'important features Correlation')



# # Train a logistic regression classifier on the training data
# clf = LogisticRegression(random_state=42)
# clf.fit(X_train, y_train)
#
# # Predict the probability of a fighter winning against an opponent
# new_fight = [[1, 3, 3, 80, 76, 32, 1, 0, 0, 0, 0, 94, 0, 1, 0, 8.7, 0, 0, 0, 0, 52, 75,1, 94]]
# prob_win = clf.predict_proba(new_fight)[:, 1]
# print('The probability of winning is:', prob_win)


#  We Store prediction of each model in our dict
# Helper Functions for our models.
# find the last elo rating and glicko rating from the dataset of the fighters and predict their probabilty of winning the match
# if the fighters are not available in the data set find the new elo rating and glicko rating
# classify the fighter based on the formula
# add or subtract the win and loose of a fight







