import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("DataSet/50_startups_sample.csv",encoding='latin1')
print(data.head())