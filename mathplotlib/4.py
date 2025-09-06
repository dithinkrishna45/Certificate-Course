import matplotlib.pyplot as ptl
import numpy as np
import pandas as pd

categories = ['A','B','C']
values = [3,7,5]

ptl.barh(categories,values,color=['purple'])
ptl.title("Horizontal Bar Plot")
ptl.show()
