import numpy as np 
'''
Implemented so far
1. Linaar Regression
2. MultiLinear Regression ( using Gradient Descent )
3. K Nearest Neighbour
4. NaiveBayesClassifer
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


class KNN:

	def __init__(self):
		pass

	def euclidDistance(self,d1,d2,len):
		dist=0
		for x in range(len):
			dist+= np.square(d1[x]-d2[x])
		return np.sqrt(dist)

	def fit(self,x,y):
		self.x=x
		self.y=y
		self.classes=list(set(y))

	def predict(self,xtest,k=1):
		y=[]
		for i in xtest:
			y.append(self.predictS(i,k))
		return np.array(y)


	def predictS(self,xtest,k=1):
		dist=[]
		for i in range(len(self.x)):
			tdist=self.euclidDistance(self.x[i],xtest,self.x.shape[1])
			dist.append((self.y[i],tdist))
		dist=sorted(dist,key=lambda x:x[1])

		topK=dist[0:k]
		classCount={}
		for classes in self.classes:
			classCount[classes]=0

		for i in topK:
			classCount[i[0]]+=1


		mx=classCount[self.classes[0]]
		label=self.classes[0]
		for i in self.classes:
			if mx < classCount[i]:
				mx=classCount[i]
				label=i

		return label

class NaiveBayesClassifer:

	def __init__(self):
		pass

	def fit(self,x,y):
		self.x=x
		self.y=y
		self.classes=list(set(y))
		self.no_features = x.shape[1]

		featureTable=[]
		self.xT = x.T
		
		g={}
		for i in range(self.no_features):
			table=[]
			tempArray={}
			plSet=list(set(self.xT[i]))

			for ele in plSet:
				classes=[0 for _ in range(len(self.classes))]
				for j in range(len(self.y)):
					classes[self.classes.index(self.y[j])]+= (1 if self.x[j][i] == ele   else 0 ) 
				tempArray[ele]=classes
			totalVals=[0 for _ in range(len(self.classes))]
			for j in tempArray:
				totalVals=[totalVals[k]+tempArray[j][k] for k in range(len(totalVals))]
			fs=[]
			for ele in tempArray:
				table.append([ tempArray[ele][k]/totalVals[k] for k in range(len(tempArray[ele])) ])

			featureTable.append(table)
			g[i]=tempArray

		self.featureTable=featureTable
		self.findX=g

	def predictS(self,xtest):
		
		classProb=[]
		for clas in range(len(self.classes)):
			res=1
			for feature in range(self.no_features):
				x = xtest[feature]
				xindex = self.findX[feature]
				xindex = list(xindex.keys()).index(x)
				res*=self.featureTable[feature][xindex][clas]
			classProb.append(res)
		t=[x/sum(classProb) for x in classProb]

		return self.classes[t.index(max(t))]


	def predict(self,xtest):
		try:
			y=[]
			for i in xtest:
				y.append(self.predictS(i))
			return y
		except:
			raise Exception("Argument not found.")


		


















class LogisticRegression:
	# takes input (x,y) where y is binary information(0,1)
	def __init__(self):
		pass

	def sigmoid(self,z):
		return 1.0/(1+np.exp(-z))

	def predict(self,ytest):
		z=np.dot(ytest,self.bias)
		return self.sigmoid(z)
	









