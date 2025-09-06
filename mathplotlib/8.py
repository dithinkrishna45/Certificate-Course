import matplotlib.pyplot as ptl
import numpy as np
import pandas as pd

data = np.random.normal(100,20,200)
ptl.boxplot(data)
ptl.title("Box Plot")
ptl.show()