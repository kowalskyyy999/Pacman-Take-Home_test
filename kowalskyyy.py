import numpy as np 

class LinearRegression(object):
	def __init__(self, learning_rate = 1e-3):
		self.lr = learning_rate

	def predict(self, X):
		_, dim = X.shape
		if dim > 1:
			## Trying for multiple regression
			return X.dot(self.weights)
		else:
			return X.dot(self.weights) + self.biases

	def cost_function(self, y_pred, y):
		error = np.power(y_pred - y, 2)
		return 1/(2*self.n) * np.sum(error)

	def fit(self, X, y):

		early_stopping = False
		best_loss = np.inf

		self.n, self.feat = X.shape

		self.weights = np.zeros(self.feat)
		self.biases = 0 
		self.X = X
		self.y = y 

		## Until Convergen
		"""
		while not early_stopping:
			self.pred = self.predict(X)
			loss = self.cost_function(self.pred, y)

			if loss < best_loss:
				best_loss = loss 
			else:
				early_stopping = True

			self.update_weights()
		"""

		for _ in range(10000):
			self.pred = self.predict(X)
			loss = self.cost_function(self.pred, y)
			if np.isnan(loss):
				break
			else:
				self.update_weights()

		return self

	def update_weights(self):
		dW = - (2 * (self.X.T.dot(self.y - self.pred))/self.n)
		db = - 2 * np.sum(self.y - self.pred)/self.n

		#dW = self.X.T.dot((self.pred - self.y))/self.n 
		#db = np.sum(self.pred - self.y)/self.n 

		self.weights = self.weights - self.lr * dW 
		self.biases = self.biases - self.lr * db

		return self


class KNeighborsRegression():
	def __init__(self, p = 1, n_neighbors = 2):
		self.k = n_neighbors
		self.p = p 

	def distance(self, a, b):
		return np.power(np.sum(np.power(abs( a - b), self.p)), 1/self.p)

	def fit(self, X, y):
		self.X_train = X
		self.y_train = y

	def predict(self, X):
		prediction = np.zeros(X.shape[0])
		for i in range(X.shape[0]):
			row = X[i]
			neighbors = self.get_neighbors(row)
			prediction[i] = np.mean(neighbors)

		return prediction

	def get_neighbors(self, X):
		distances = np.zeros(self.X_train.shape[0])
		for i, row in enumerate(self.X_train):
			distances[i] = self.distance(X, row)

		inds = distances.argsort()
		y_train = self.y_train[inds]
		return y_train[:self.k]

class PCA():
	def __init__(self, n_components):
		self.n_components = n_components
		self.components = None 
		self.mean = None 

	def fit(self, X):
		self.mean = np.mean(X, axis=0)
		# X -= self.mean
		X_ = X - self.mean
		#sigma = 1/(X.shape[0]-1) * X.T.dot(X)
		sigma = 1/(X.shape[0] - 1) * X_.T.dot(X_)
		U, s, Vt = np.linalg.svd(X_)
		self.components = Vt.T[:, :self.n_components]

	def transform(self, X):
		X_ = X - self.mean 
		return X_.dot(self.components)

	def fit_transform(self, X):
		self.fit(X)
		return self.transform(X)



