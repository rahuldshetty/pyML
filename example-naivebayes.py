from models import *
import numpy as np 

x=np.array([[1,0,0],[2,1,0],[2,0,1],[2,0,0],[2,1,1],[1,0,1],[1,1,1]])
y=np.array([1,0,0,0,0,1,1])

model=NaiveBayesClassifer()
model.fit(x,y)

ytest=np.array([[1,0,0],[2,1,1],[2,0,1],[1,1,1]])
print(model.predict(ytest))