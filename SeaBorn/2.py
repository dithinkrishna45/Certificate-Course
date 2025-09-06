import seaborn as sns
import pandas as pd
import matplotlib.pyplot as ptl

data = pd.DataFrame({
    "Category":['A','B','C','D'],
    "Value":[4,7,2,9]
})

sns.barplot(x="Category",y="Value",data=data)
ptl.title("Normal Bar Chart (SeaBorn)")
ptl.show()