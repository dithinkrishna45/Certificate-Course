import pandas as pd

data = pd.read_csv("Certificate-Course/DataSet/MISSING_DATASET_HANDLING.csv",encoding='latin1')
print(data.isnull().sum())