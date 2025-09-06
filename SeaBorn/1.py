import matplotlib.pyplot as ptl
import seaborn as sns
df = sns.load_dataset("tips")
sns.histplot(df["total_bill"],bins=20,kde=True,color='orange')
ptl.title("Histogram + KDE")
ptl.show()
