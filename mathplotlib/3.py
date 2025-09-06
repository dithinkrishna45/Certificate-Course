import matplotlib.pyplot as ptl
import numpy as np
import pandas as pd

categories = ['A','B','C']
values = [3,7,5]

ptl.bar(categories,values,color=['red','green','blue'])
ptl.title("Bar Plot")
ptl.show()
