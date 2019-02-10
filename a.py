from models import *
import numpy as np 

x=np.array([1,2,3,4,5,6,7,8,9])
y=np.array([1,4,9,16,25,36,49,64,81])

model=LinearRegression()
model.fit(x,y)

ytest=np.array([15,17,22])
print(model.predict(ytest))