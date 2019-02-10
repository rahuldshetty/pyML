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
	def __init__(self,lr=0.0001,iterations=1000):
		self.lr=lr
		self.iterations=iterations
	def cost_function(self,X,Y,B):
		m=len(Y)
		J=np.sum((X.dot(B)-Y)**2)/(2*m)
		return J

	def print(self):
		print("Cost History:")
		print("Starting:",self.cost_history[0])
		print("After",self.iterations,":",self.cost_history[len(self.cost_history)-1])


	def gradient_descent(self,X,Y,B):
		cost_history=[0]*self.iterations
		m=len(Y)
		for iterations in range(self.iterations):
			h=X.dot(B)
			loss=h-Y
			gradient=X.T.dot(loss)/m
			B=B-self.lr * gradient

			cost=self.cost_function(X,Y,B)
			cost_history[iterations]=cost
		return B,cost_history

	def fit(self,x,y):
		self.x=x
		self.y=y
		cols=x.shape[1]
		rows=x.shape[0]
		x0=np.ones(rows)
		X=x
		B=np.zeros(cols)

		initial_cost = self.cost_function(X,y,B)
		newB,cost_history = self.gradient_descent(X,y,B)
		self.cost_history=cost_history
		self.bias=newB


	def predict(self,x):
		return self.bias.T*x









