import matplotlib.pyplot as ptl
import numpy as np
import pandas as pd

sizes = [20,30,25,25]
labels = ['A','B','C','D']
ptl.pie(sizes,labels=labels,autopct='%1.1f%%',startangle=90)
ptl.title("Pie Chart")
ptl.show()