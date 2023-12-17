import numpy as np
from itertools import product
import time

levi = np.zeros((3,3,3))
levi[0,1,2], levi[1,2,0] ,levi[2,0,1] = 1, 1, 1
levi[1,0,2], levi[2,1,0] ,levi[0,2,1] = -1, -1, -1