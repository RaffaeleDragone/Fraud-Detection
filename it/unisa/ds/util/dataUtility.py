#fagov83809@whowlft.com: Password_123
import os

#import kaggle
import kaggle
import pandas as pd
from sklearn import preprocessing

pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


def download_and_unzip_dataset():
    print("download")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('ntnu-testimon/paysim1', path='./datasets', unzip=True)

def load_paysim_data():
    csv_path = os.path.join('../dataset/', 'PS_20174392719_1491204439457_log.csv')
    return pd.read_csv(csv_path)

def information_dataframe(dataFrame):
    print(dataFrame.info())
    print('Tipi di transazioni presenti \n',dataFrame['type'].value_counts())
    print('\n Il Tipo di transazioni che risultano essere fraudolente nel dataset sono {} '.format( \
    list(dataFrame.loc[dataFrame.isFraud == 1].type.drop_duplicates().values)))

    print('Transazioni Fraudolente', len(dataFrame.loc[(dataFrame.isFraud==1)]))
    print('Transazioni Fraudolente TRANSFER', len(dataFrame.loc[(dataFrame.isFraud==1) & (dataFrame.type == 'TRANSFER')]))
    print('Transazioni Fraudolente CASH_OUT', len(dataFrame.loc[(dataFrame.isFraud == 1) & (dataFrame.type == 'CASH_OUT')]))

    print('Transazioni Flagged', len(dataFrame.loc[(dataFrame.isFlaggedFraud == 1)]))

    print('\nThe type of transactions in which isFlaggedFraud is set: \ {}'.format(
        list(dataFrame.loc[dataFrame.isFlaggedFraud == 1].type.drop_duplicates())))





    #Nella guida fa vedere come isFlaggedFraud non è correlato con nessun altro campo. Quindi dice che poiché tra l'altro ci sono solo 16 record con tale valore lo scarta
    #Decidiamo poi se metterci questo ragionamento


def information_numeric_data(dataFrame):
    print(dataFrame.describe())

def normalize_dataset(df):
    X = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]

    Y = X['isFraud']

    del X['isFraud']

    # Eliminate columns shown to be irrelevant for analysis in the EDA
    X = X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)


    return X
    #Ci sono altre operazioni che possono essere effettuate. Ho trovate alcune discussioni :
    #https://www.kaggle.com/netzone/eda-and-fraud-detection
    #https://www.kaggle.com/stark10war/fraud-detection-eda-and-modelling
    #https://www.kaggle.com/aadilrafeeque/exploratory-data-analysis-and-fraud-prediction
    #Diciamo che quelle che hanno nel titolo "EDA" fanno proprio focus sulla pulizia dei dati


def feature_selection (df):
    # Occorrenze delle varie features per le transazioni che risultano essere fraudolente
    # STEP
    step_list = list(df.loc[df.isFraud == 1].step.values)
    step_counted_list = Counter(step_list)
    print(step_counted_list.most_common(20), '\n')

    # TYPE
    type_list = list(df.loc[df.isFraud == 1].type.values)
    type_counted_list = Counter(type_list)
    print(type_counted_list.most_common(20), '\n')

     #AMOUNT
    amount_list = list(df.loc[df.isFraud == 1].amount.values)
    amount_counted_list = Counter(amount_list)
    print(amount_counted_list.most_common(20), '\n')

    #NAME ORIGIN
    nameOrig_list = list(df.loc[df.isFraud == 1].nameOrig.values)
    nameOrig_counted_list = Counter(nameOrig_list)
    print(nameOrig_counted_list.most_common(20), '\n')

    #OLD_BALANCE_ORIGIN
    oldBalanceOrig_list = list(df.loc[df.isFraud == 1].oldbalanceOrg.values)
    oldBalanceOrig_counted_list = Counter(oldBalanceOrig_list)
    print(oldBalanceOrig_counted_list.most_common(20), '\n')

    #NEW BALANCE ORIGIN
    newBalanceOrig_list = list(df.loc[df.isFraud == 1].newbalanceOrig.values)
    newBalanceOrig_counted_list = Counter(newBalanceOrig_list)
    print(newBalanceOrig_counted_list.most_common(20), '\n')

    #NAME DEST
    nameDest_list = list(df.loc[df.isFraud == 1].nameDest.values)
    nameDest_counted_list = Counter(nameDest_list)
    print(nameDest_counted_list.most_common(20), '\n')

    #OLD BALANCE DEST
    oldBalanceDest_list = list(df.loc[df.isFraud == 1].oldbalanceDest.values)
    oldBalanceDest_counted_list = Counter(oldBalanceDest_list)
    print(oldBalanceDest_counted_list.most_common(20), '\n')

    #NEW BALANCE DEST
    newBalanceDest_list = list(df.loc[df.isFraud == 1].newbalanceDest.values)
    newBalanceDest_counted_list = Counter(newBalanceDest_list)
    print(newBalanceDest_counted_list.most_common(20), '\n')

    #IS_FLAGGED
    isFlaggedFraud_list = list(df.loc[df.isFraud == 1].isFlaggedFraud.values)
    isFlaggedFraud_counted_list = Counter(isFlaggedFraud_list)
    print(isFlaggedFraud_counted_list.most_common(20), '\n')

    '''Features:

1. step : Include this feature. The fraudulent transactions distributed in many 'step' values.

2. type : Include this feature. The fraudulent transaction happened only in 'CASH_OUT' and 'TRANSFER' transaction types. So we will include only the records with type as 'CASH_OUT' and 'TRANSFER.'

3. amount: Include this feature. Though it won't explain all fraudulent transactions, amount as 10000000.0 and 0.0 denotes a high chance of fraud.

4. nameOrig : Drop this feature. There is no useful information from this column.

5. oldbalanceOrig : Include this feature. You could see that in almost all fraudulent transactions, 'oldbalanceOrig' and 'amount' has the same value. This is a strong indicator of a fraudulent transaction.

6. newbalanceOrig : Include this feature. For most of the fraudulent transactions, 'newbalanceOrig' = 0 (this fact supports our finding in #5)

7. nameDest : Drop this feature. There is no useful information from this column.

8. oldbalanceDest : Include this feature. Value of 'oldbalanceDest' is zero for nearly half of the fraudulent transaction.

9. newbalanceDest : Include this feature. Value of 'oldbalanceDest' is zero for more than half of the fraudulent transaction. We will include this feature in our model.

10. isFlaggedFraud : Drop this feature. Only 16 transactions flagged correctly. We can drop this feature.'''



def dataCleaning(df):
    X = df.loc[(df.type == 'CASH_OUT') | (df.type == 'TRANSFER')]
    randomState = 5
    np.random.seed(randomState)
    Y = X['isFraud']
    del X['isFraud']

    del X['nameDest']
    del X['nameOrig']
    del X['isFlaggedFraud']
    X.loc[X.type == 'TRANSFER', 'type'] = 0
    X.loc[X.type == 'CASH_OUT', 'type'] = 1
    X.type = X.type.astype(int)
    #print(X.head())
    X_fraud = X.loc[Y == 1]
    X_nonFraud = X.loc[Y == 0]

    Xfraud = X.loc[Y == 1]
    XnonFraud = X.loc[Y == 0]
    print('\nLa percentuale di transazioni fraudolente con \'oldBalanceDest\' = \
    \'newBalanceDest\' = 0 con una transazione con \'amount\' diverso da 0 è: {}'. \
          format(len(Xfraud.loc[(Xfraud.oldbalanceDest == 0) & \
                                (Xfraud.newbalanceDest == 0) & (Xfraud.amount)]) / (1.0 * len(Xfraud))))

    print('\nLa percentuale di transazioni non fraudolente con \'oldBalanceDest\' = \
    newBalanceDest\' = 0 con una transazione con \'amount\' diverso da 0 è: {}'. \
          format(len(XnonFraud.loc[(XnonFraud.oldbalanceDest == 0) & \
                                   (XnonFraud.newbalanceDest == 0) & (XnonFraud.amount)]) / (1.0 * len(XnonFraud))))
    dfOutDiz = {}
    dfOutDiz[0]=X
    dfOutDiz[1]=X_fraud
    dfOutDiz[2]=X_nonFraud
    dfOutDiz[3]=Y
    return dfOutDiz


def returnDataWithoutCleaning(df):
    X = df
    randomState = 5
    np.random.seed(randomState)
    Y = X['isFraud']
    del X['isFraud']

   # del X['nameDest']
   # del X['nameOrig']
   # del X['isFlaggedFraud']
    X.loc[X.type == 'TRANSFER', 'type'] = 0
    X.loc[X.type == 'CASH_OUT', 'type'] = 1
    X.loc[X.type == 'PAYMENT', 'type'] = 2
    X.loc[X.type == 'CASH_IN', 'type'] = 3
    X.loc[X.type == 'DEBIT', 'type'] = 4
    #X.type = X.type.astype(int)
    le = preprocessing.LabelEncoder()
    X["nameOrig"]= le.fit_transform(X["nameOrig"])
    X["nameDest"] = le.fit_transform(X["nameDest"])
    # print(X.head())
    X_fraud = X.loc[Y == 1]
    X_nonFraud = X.loc[Y == 0]

    dfOutDiz = {}
    dfOutDiz[0] = X
    dfOutDiz[1] = X_fraud
    dfOutDiz[2] = X_nonFraud
    dfOutDiz[3] = Y
    return dfOutDiz

def dataCleaningAndEngineering(df):
    X = df.loc[(df.type == 'CASH_OUT') | (df.type == 'TRANSFER')]


    Y = X['isFraud']
    del X['isFraud']

    del X['nameDest']
    del X['nameOrig']
    del X['isFlaggedFraud']

    X['errorBalanceOrig'] = X.newbalanceOrig + X.amount - X.oldbalanceOrg
    X['errorBalanceDest'] = X.oldbalanceDest + X.amount - X.newbalanceDest

    X.loc[X.type == 'TRANSFER', 'type'] = 0
    X.loc[X.type == 'CASH_OUT', 'type'] = 1
    X.type = X.type.astype(int)
    #print(X.head())

    '''
        corr_matrix = X.corr()
        corr_matrix["isFraud"].sort_values(ascending=False)
        print(corr_matrix)
    '''
    print(X)
    X_fraud = X.loc[Y == 1]
    X_nonFraud = X.loc[Y == 0]

    dfOutDiz = {}
    dfOutDiz[0]=X
    dfOutDiz[1]=X_fraud
    dfOutDiz[2]=X_nonFraud
    dfOutDiz[3]=Y
    return dfOutDiz



dataFrame = load_paysim_data()

#information_dataframe(dataFrame)
#information_numeric_data(dataFrame)

dataCleaning(dataFrame)
#dataCleaning(dataFrame)
#df_normalized = normalize_dataset(dataFrame)

#print(df_normalized.head(10))


#dataFrame.hist(bins=50, figsize=(20,15))
#plt.show()

#Fase di Feature Selection che mi sembra fatta bene e anche con ragionamenti non da astrologi :
#https://www.kaggle.com/georgepothur/4-financial-fraud-detection-xgboost

#dataCleaningAndEngineering(dataFrame)

