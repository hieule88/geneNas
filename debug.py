import numpy as np
xc = np.array([ 1.,0.,0.,0.,0.])

zeor = np.zeros((5,1))
print(xc.shape)
print(zeor.shape)

print(np.all(zeor==xc))