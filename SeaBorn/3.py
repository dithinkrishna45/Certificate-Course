import seaborn as sns
import matplotlib.pyplot as ptl

df = sns.load_dataset("tips")
sns.countplot(x='day',data=df,palette='Set3')
ptl.title('Count Plot')
ptl.show()