import matplotlib.pyplot as ptl
import numpy as np
import pandas as pd

data = np.random.randn(1000)
ptl.hist(data,bins=100,color='black',edgecolor='white')
ptl.title('Histogram')
ptl.show()