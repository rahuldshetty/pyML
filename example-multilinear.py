from models import *
import numpy as np 

x=np.array([[1],[2],[3],[4]])
y=np.array([2,3,4,5])

model=MultiLinearRegression(iterations=10000)
model.fit(x,y)

ytest=np.array([[9]])
print(model.predict(ytest))