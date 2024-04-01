import numpy as np

a = [[1,2,3], [1,2,3], [1,2,3], [1,2,3]]
b = [[4,5], [4,5], [4,5], [4,5]]
print(np.concatenate((a, b), axis=1))