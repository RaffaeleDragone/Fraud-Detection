#fagov83809@whowlft.com: Password_123
import os

#import kaggle
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.5f' % x)
import numpy as np
import matplotlib.pyplot as plt



def download_and_unzip_dataset():
    print("download")
    #kaggle.api.authenticate()
    #kaggle.api.dataset_download_files('ntnu-testimon/paysim1', path='./datasets', unzip=True)

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
        list(dataFrame.loc[dataFrame.isFlaggedFraud == 1].type.drop_duplicates())))  # only 'TRANSFER'

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

    #Ci sono altre operazioni che possono essere effettuate. Ho trovate alcune discussioni :
    #https://www.kaggle.com/netzone/eda-and-fraud-detection
    #https://www.kaggle.com/stark10war/fraud-detection-eda-and-modelling
    #https://www.kaggle.com/aadilrafeeque/exploratory-data-analysis-and-fraud-prediction
    #Diciamo che quelle che hanno nel titolo "EDA" fanno proprio focus sulla pulizia dei dati



    return X

dataFrame = load_paysim_data()

information_dataframe(dataFrame)
#information_numeric_data(dataFrame)


df_normalized = normalize_dataset(dataFrame)

print(df_normalized.head(10))


dataFrame.hist(bins=50, figsize=(20,15))
plt.show()

#Fase di Feature Selection che mi sembra fatta bene e anche con ragionamenti non da astrologi :
# https://www.kaggle.com/georgepothur/4-financial-fraud-detection-xgboost


