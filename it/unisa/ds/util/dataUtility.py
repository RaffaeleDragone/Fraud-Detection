#fagov83809@whowlft.com: Password_123
import os
import pandas as pd
import matplotlib.pyplot as plt
def load_paysim_data():
    csv_path= os.path.join('../dataset/', 'PS_20174392719_1491204439457_log.csv')
    return pd.read_csv(csv_path)


dataFrame = load_paysim_data()
#print(dataFrame.head())
print(dataFrame.info())
print("*********** type attribute ")
print(dataFrame["type"].value_counts())
print("*********** nameOrig attribute ")
print(dataFrame["nameOrig"].value_counts())
print("*********** nameDest attribute ")
print(dataFrame["nameDest"].value_counts())

dataFrame.hist(bins=50, figsize=(20,15))
plt.show()
#pd.set_option('display.max_columns', 500)
#pd.set_option('display.float_format', lambda x: '%.3f' % x)
#print(dataFrame.describe())



