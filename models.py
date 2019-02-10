import numpy as np 
'''
Implemented so far
1. Linaar Regression
'''
class LinearRegression:
	#1 dimensional y=mx+c
	def __init__(self):
		pass
	def print(self):
		print("Mean X:",self.x_mean)
		print("Mean Y:",self.y_mean)
		print("Expression: Y = M*X + C ")
		print("M - Slope")
		print("C - Intercept")
	def fit(self,x,y):
		self.x=x
		self.y=y
		x_mean=np.mean(x)
		y_mean=np.mean(y)
		self.x_mean=x_mean
		self.y_mean=y_mean
		n=len(x)
		num = 0
		den = 0
		for i in range(n):
			num+=(x[i]-x_mean)*(y[i]-y_mean)
			den+=(x[i]-x_mean)**2
		m=num/den
		c=y_mean - (m*x_mean)
		self.m=m
		self.c=c
	def predict(self,ytest):
		y=[]
		for i in ytest:
			y.append(self.m*i + self.c )
		return np.array(y)

class MultiLinearRegression:
	def __init__(self):
		pass
	def fit(self,x,y):
		self.x=x
		self.y=y
