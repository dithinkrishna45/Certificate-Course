import matplotlib.pyplot as ptl
import numpy as np
import pandas as pd

days = [1,2,3,4,5]
sleeping = [7,8,6,11,7]
eating = [2,3,4,3,2]
working = [7,8,7,2,2]
playing = [8,5,7,8,13]

ptl.stackplot(days,sleeping,eating,working,playing,labels=['Sleep','Eat','Work','Play'])
ptl.legend(loc='upper left')
ptl.title("Stacked Area Plot")
ptl.show()