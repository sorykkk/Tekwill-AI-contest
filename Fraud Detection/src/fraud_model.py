import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score
from xgboost.sklearn import XGBClassifier
from ML_features import *
from data_analisys import *
from data_visualization import *

import gdown
import warnings
import os.path
warnings.filterwarnings("ignore")
#warnings.filterwarnings("ignore", category=DeprecationWarning)


url = "https://drive.google.com/file/d/1TDRgAS-dhx8B4Gu6GAdOpIlEybZd6idv/view?usp=sharing"
output = "./input/PS_20174392719_1491204439457_log.csv"

if not os.path.isfile("./input/PS_20174392719_1491204439457_log.csv"):
    gdown.download(url=url, output=output, quiet=False, fuzzy = True)

df = correct_data(pd.read_csv("./input/PS_20174392719_1491204439457_log.csv"))

### Analiza datelor ###

#types_of_fraud(df)
#isFlaggedFraud_info(df)
#merchant_info(df)

# selectam doar coloanele unde sunt prezente fraudele (in umra analizei)
X = df.loc[(df.type == "TRANSFER") | (df.type == "CASH_OUT")]

random_state = 5
np.random.seed(random_state)

Y = X["isFraud"]
del X["isFraud"]

# eliminam coloanele care au dovedit a fi irelevante in urma analizei
X = X.drop(["nameOrig", "nameDest", "isFlaggedFraud"], axis = 1)

# codificare binară a datelor etichetate în "type"
X.loc[X.type == "TRANSFER", "type"] = 0
X.loc[X.type == "CASH_OUT", "type"] = 1
X.type = X.type.astype(int)

X_fraud = X.loc[Y == 1]
X_non_fraud = X.loc[Y == 0]

# data are multe tranzactii cu balanta 0 la contul de destinatie inainte si dupa ce un amount diferit de 0 a fost trimis
# rata acestor tranzactii, unde 0 posibil denota lipsa valorii, este mai mare la cele frauduloase (50%) comparate cu cele adevarate (0.06%)
print("\nRata tranzactiilor fraudulente cu oldBalanceDest = newBalanceDest = 0 chiar daca s-a tranzactionat un amount diferit de 0 este: {}" 
    .format(len(X_fraud.loc[(X_fraud.oldBalanceDest == 0) & (X_fraud.newBalanceDest == 0) & (X_fraud.amount)]) / (1.0 * len(X_fraud))))

print("\nRata tranzactiilor reale cu oldBalanceDest = newBalanceDest = 0 chiar daca s-a tranzactionat un amount non-zero este: {}" 
    .format(len(X_non_fraud.loc[(X_non_fraud.oldBalanceDest == 0) & (X_non_fraud.newBalanceDest == 0) & (X_non_fraud.amount)]) / (1.0 * len(X_non_fraud))))

# balanta contului destinatar 0 este un indicator puternic a fraudei, 
# nu imputam soldul (balanta) contului (inainte de efectuarea tranzactiei) cu o statistica 
# inlocuim valoarea 0 cu -1, ceea ce va fi mai util pentru un algoritm adecvat de invatare automata care detecteaza frauda
X.loc[(X.oldBalanceDest == 0) & (X.newBalanceDest == 0) & (X.amount != 0), ["oldBalanceDest", "newBalanceDest"]] = - 1
# data de asemenea are multe tranzactii cu soldul 0 in contul de origine,
# inainte si dupa ce o valoare nonzero a fost tranzactionata.
# din aceleasi motive ca in cazul descris mai sus, in loc sa imputam o valoare numerica noi inlocuim valoare de 0 cu valoarea null
X.loc[(X.oldBalanceOrig == 0) & (X.newBalanceOrig == 0) & (X.amount != 0), ["oldBalanceOrig", "newBalanceOrig"]] = np.nan

# mai cream 2 coloane care vor inregistra errorile in conturile originare (initiante) si destinatare pentru fiecare tranzactie
X["errorBalanceOrig"] = X.newBalanceOrig + X.amount - X.oldBalanceOrig
X["errorBalanceDest"] = X.oldBalanceDest + X.amount - X.newBalanceDest

### Vizualizarea datelor ###

#dispersion_over_time(X, Y)
#dispersion_over_amount(X, Y)
#dispersion_over_errorBalanceDest(X, Y)
#separate_transactions(X, Y)
#fingerprint_transactions(X, Y)

# verificam cat e de deformata data
print(f"skew = { len(X_fraud) / float(len(X))}")
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.2, random_state = random_state)
# s-a dovedit a fi destul de deformata, de accea folosim AUPRC in loc de AUROC


# compilare destul de lunga 
weights = (Y == 0).sum() / (1.0 * (Y == 1).sum())
# folosim xgboost pentru a ne descurca cu lipsele de valori din data si pentru a ajuta la
# accelerare prin prcesarea in parale, intrecand random-forest
classifier = XGBClassifier(max_depth = 3, scale_pos_weight = weights, n_jobs = 4)

probabilities = classifier.fit(train_X, train_Y).predict_proba(test_X)
print(f"AUPRC = {average_precision_score(test_Y, probabilities[:, 1])}")

# model_name = "fraud_prediction_model.json"
# classifier.save_model("src/"+model_name)

#features_importance(classifier)

#train_sizes, train_scores, cross_val_scores = learning_curve(\
#                                            XGBClassifier(max_depth = 3, scale_pos_weight = weights, n_jobs = 4),\
#                                             train_X,train_Y, scoring = "average_precision")

#bias_variance_tradeoff(train_sizes, train_scores, cross_val_scores)