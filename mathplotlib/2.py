import matplotlib.pyplot as ptl
import numpy as np
import pandas as pd

x=np.random.rand(50)
y=np.random.rand(50)

ptl.scatter(x,y,color='orange',marker='*')
ptl.title('Scatter Plot')
ptl.show()