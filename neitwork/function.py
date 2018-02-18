import numpy as np

def logistic_sigmoid(x):
	return 1 / (1 + np.exp(-x))
def d_logistic_sigmoid(x):
	return 1 / (np.exp(0.5 * x) + np.exp(-0.5 * x)) ** 2

def tanh(x):
	return np.tanh(x)
def d_tanh(x):
	return 1 / np.cosh(x) ** 2

def ramp(x):
	return np.maximum(0, x)
def d_ramp(x):
	return 1 * (x > 0)

def ident(x):
	return x
def d_ident(x):
	return np.ones_like(x)

def soft_max(x):
	c = np.max(x, axis = 1, keepdims = True)
	exp = np.exp(x - c)
	sum_exp = np.sum(exp, axis = 1, keepdims = True)
	return exp / sum_exp
def d_soft_max(x):
	y = soft_max(x)
	N = y.shape[0]
	df = -y.reshape(N, -1, 1) * y.reshape(N, 1, -1)
	df += np.apply_along_axis(np.diag, 1, y)
	return df

def mean_squared_error(y, d):
	return 0.5 * np.sum((y - d) ** 2, axis = 1)
def d_mean_squared_error(y, d):
	return y - d

def cross_entropy_error(y, d):
	delta = 1e-7
	return -np.sum(d * np.log(y + delta), axis = 1)
def d_cross_entropy_error(y, d):
	return -d / y

def he_init_std_dev(n):
	return np.sqrt(2 / n)
def xavier_init_std_dev(n):
	return 1 / np.sqrt(n)

