#fagov83809@whowlft.com: Password_123
import os

import kaggle
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.5f' % x)
import numpy as np
import matplotlib.pyplot as plt



def download_and_unzip_dataset():
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

def information_numeric_data(datFrame):
    print(dataFrame.describe())

dataFrame = load_paysim_data()

#information_dataframe(dataFrame)
information_numeric_data(dataFrame)





#dataFrame.hist(bins=50, figsize=(20,15))
#plt.show()




