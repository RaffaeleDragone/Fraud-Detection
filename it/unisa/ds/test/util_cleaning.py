import os
import pandas as pd
import numpy as np
from collections import Counter

from sklearn import preprocessing


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


def step_1(df_orig):
    df = df_orig.copy()
    print('********* Returning origin dataset with only transaction typed as CASH_OUT or TRANSFER . Converted them to 0 and 1 *************')

    # Occorrenze delle varie features per le transazioni che risultano essere fraudolente
    # TYPE
    type_list = list(df.loc[df.isFraud == 1].type.values)
    type_counted_list = Counter(type_list)
    print(type_counted_list.most_common(20), '\n')

    print('Transazioni Fraudolente', len(df.loc[(df.isFraud==1)]))
    print('Transazioni Fraudolente TRANSFER', len(df.loc[(df.isFraud==1) & (df.type == 'TRANSFER')]))
    print('Transazioni Fraudolente CASH_OUT', len(df.loc[(df.isFraud == 1) & (df.type == 'CASH_OUT')]))

    X = df.loc[(df.type == 'CASH_OUT') | (df.type == 'TRANSFER')]
    X.loc[X.type == 'TRANSFER', 'type'] = 0
    X.loc[X.type == 'CASH_OUT', 'type'] = 1
    X.type = X.type.astype(int)

    Y = X['isFraud']
    del X['isFraud']

    dfOutDiz = {}
    dfOutDiz[0] = X
    dfOutDiz[1] = Y
    return dfOutDiz

def step_2(df_orig):
    df = df_orig.copy()
    print('********* Returning origin dataset with only transaction typed as CASH_OUT or TRANSFER . Converted them to 0 and 1 \n '
          ' Removed attribute IsFlaggedFraud *************')
    # Occorrenze delle varie features per le transazioni che risultano essere fraudolente
    # IS_FLAGGED
    isFlaggedFraud_list = list(df.loc[df.isFraud == 1].isFlaggedFraud.values)
    isFlaggedFraud_counted_list = Counter(isFlaggedFraud_list)
    print(isFlaggedFraud_counted_list.most_common(20), '\n')

    X = df.loc[(df.type == 'CASH_OUT') | (df.type == 'TRANSFER')]
    X.loc[X.type == 'TRANSFER', 'type'] = 0
    X.loc[X.type == 'CASH_OUT', 'type'] = 1
    X.type = X.type.astype(int)

    del X['isFlaggedFraud']

    le = preprocessing.LabelEncoder()
    X['nameOrig'] = le.fit_transform(X["nameOrig"])
    X['nameDest'] = le.fit_transform(X["nameDest"])

    Y = X['isFraud']
    del X['isFraud']

    dfOutDiz = {}
    dfOutDiz[0] = X
    dfOutDiz[1] = Y
    return dfOutDiz


def step_3(df_orig):
    df = df_orig.copy()
    print(
        '********* Returning origin dataset with only transaction typed as CASH_OUT or TRANSFER . Converted them to 0 and 1 \n '
        ' Removed attribute IsFlaggedFraud \n '
        ' Removed attribute NameOrig *************')
    # Occorrenze delle varie features per le transazioni che risultano essere fraudolente
    # NAME ORIGIN
    nameOrig_list = list(df.loc[df.isFraud == 1].nameOrig.values)
    nameOrig_counted_list = Counter(nameOrig_list)
    print(nameOrig_counted_list.most_common(20), '\n')

    X = df.loc[(df.type == 'CASH_OUT') | (df.type == 'TRANSFER')]
    X.loc[X.type == 'TRANSFER', 'type'] = 0
    X.loc[X.type == 'CASH_OUT', 'type'] = 1
    X.type = X.type.astype(int)

    del X['isFlaggedFraud']


    del X['nameOrig']

    le = preprocessing.LabelEncoder()
    X['nameDest'] = le.fit_transform(X["nameDest"])

    Y = X['isFraud']
    del X['isFraud']

    dfOutDiz = {}
    dfOutDiz[0] = X
    dfOutDiz[1] = Y
    return dfOutDiz

def step_4(df_orig):
    df = df_orig.copy()
    print(
        '********* Returning origin dataset with only transaction typed as CASH_OUT or TRANSFER . Converted them to 0 and 1 \n '
        ' Removed attribute IsFlaggedFraud \n '
        ' Removed attribute NameOrig \n'
        ' Removed attribute NameDest *************')
    # Occorrenze delle varie features per le transazioni che risultano essere fraudolente
    # NAME DEST
    nameDest_list = list(df.loc[df.isFraud == 1].nameDest.values)
    nameDest_counted_list = Counter(nameDest_list)
    print(nameDest_counted_list.most_common(20), '\n')

    X = df.loc[(df.type == 'CASH_OUT') | (df.type == 'TRANSFER')]
    X.loc[X.type == 'TRANSFER', 'type'] = 0
    X.loc[X.type == 'CASH_OUT', 'type'] = 1
    X.type = X.type.astype(int)

    del X['isFlaggedFraud']
    del X['nameOrig']
    del X['nameDest']

    Y = X['isFraud']
    del X['isFraud']

    dfOutDiz = {}
    dfOutDiz[0] = X
    dfOutDiz[1] = Y
    return dfOutDiz

def step_5(df_orig):
    df = df_orig.copy()
    print(
        '********* Returning origin dataset with only transaction typed as CASH_OUT or TRANSFER . Converted them to 0 and 1 \n '
        ' Removed attribute IsFlaggedFraud \n '
        ' Removed attribute NameOrig \n'
        ' Removed attribute NameDest \n '
        ' Added two new features : errorBalanceOrig and errorBalanceDest *************')
    X = df.loc[(df.type == 'CASH_OUT') | (df.type == 'TRANSFER')]
    X.loc[X.type == 'TRANSFER', 'type'] = 0
    X.loc[X.type == 'CASH_OUT', 'type'] = 1
    X.type = X.type.astype(int)

    del X['isFlaggedFraud']
    del X['nameOrig']
    del X['nameDest']

    X['errorBalanceOrig'] = X.newbalanceOrig + X.amount - X.oldbalanceOrg
    X['errorBalanceDest'] = X.oldbalanceDest + X.amount - X.newbalanceDest

    Y = X['isFraud']
    del X['isFraud']

    dfOutDiz = {}
    dfOutDiz[0] = X
    dfOutDiz[1] = Y
    return dfOutDiz

def simple_df(df_orig):
    df = df_orig.copy()
    X = df
    randomState = 5
    np.random.seed(randomState)
    Y = X['isFraud']
    del X['isFraud']


    X.loc[X.type == 'TRANSFER', 'type'] = 0
    X.loc[X.type == 'CASH_OUT', 'type'] = 1
    X.loc[X.type == 'PAYMENT', 'type'] = 2
    X.loc[X.type == 'CASH_IN', 'type'] = 3
    X.loc[X.type == 'DEBIT', 'type'] = 4


    X_fraud = X.loc[Y == 1]
    X_nonFraud = X.loc[Y == 0]

    le = preprocessing.LabelEncoder()
    X['nameOrig'] = le.fit_transform(X["nameOrig"])
    X['nameDest'] = le.fit_transform(X["nameDest"])
    # convert string values to ascii - SOSTITUISCI COL TUO CODICE
    #for index, row in df.iterrows():
    #    row['nameOrig'] = ''.join(str(ord(c)) for c in row['nameOrig'])
    #    row['nameDest'] = ''.join(str(ord(c)) for c in row['nameDest'])


    dfOutDiz = {}
    dfOutDiz[0] = X
    dfOutDiz[1] = Y
    return dfOutDiz