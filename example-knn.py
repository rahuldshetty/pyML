from models import *
import numpy as np 

x=np.array([[1],[2],[3],[4]])
y=np.array([1,1,5,5])

model=KNN()
model.fit(x,y)

ytest=np.array([[6],[-1]])
print(model.predict(ytest,k=3))