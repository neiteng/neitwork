import numpy as np
from . import function as func
from . import util
import time
from abc import ABCMeta, abstractmethod

class abstract_layer(metaclass = ABCMeta):
	@abstractmethod
	def forward(self, x):
		pass

	@abstractmethod
	def backward(self, dy):
		pass

'''
learning_layer have to update own weight in order to learn
'''
class learning_layer(abstract_layer, metaclass = ABCMeta):
	'''
	ret : dict of grad of params
	'''
	@abstractmethod
	def get_grad(self):
		pass

	@abstractmethod
	def get_weight(self):
		pass
	'''
	delta : dict of change of params
	'''
	@abstractmethod
	def update_weight(self, delta):
		pass

class linear_layer(learning_layer):
	def __init__(self, W, b):
		self.W = W
		self.b = b
		self.dW = None
		self.db = None
		self.x = None

	def forward(self, x):
		self.x = x
		return np.dot(x, self.W) + self.b

	def backward(self, dy):
		n  = dy.shape[0]
		self.dW = np.dot(self.x.T, dy) / n
		self.db = np.mean(dy, axis = 0)
		return np.dot(dy, self.W.T)

	def get_weight(self):
		return { "W" : self.W, "b" : self.b }

	def get_grad(self):
		return { "W" : self.dW, "b" : self.db }

	def update_weight(self, delta):
		self.W += delta["W"]
		self.b += delta["b"]

class convolution_layer(learning_layer):
	'''
	shape : list of int (input height, input width, num of input channel, num of output channel)
	filter_shape : list of int (filter height, filter width)
	stride : list of int (vertical, horizontal)
	pad : list of int (top, botom, left, right)
	'''
	# todo : padding_mode
	# (k, fh, fw) * m, m : next channel num
	def __init__(self, W, b, shape, filter_shape, stride, pad, padding_mode = "constant", **padding_args):
		self.W = W
		self.b = b
		self.dW = None
		self.db = None
		self.x = None
		self.h = shape[0]
		self.w = shape[1]
		self.k = shape[2]
		self.m = shape[3]
		self.fh = filter_shape[0]
		self.fw = filter_shape[1]
		self.stride = stride
		self.pad = pad
		self.padding_mode = padding_mode
		self.padding_args = padding_args
		self.oh = (self.h + pad[0] + pad[1] - self.fh) // stride[0] + 1
		self.ow = (self.w + pad[2] + pad[3] - self.fw) // stride[1] + 1

	# x : n * (h, w, k)
	# ret : n * (oh, ow, m)
	# correct 2/18, 1:38
	def forward(self, x):
		n = x.shape[0]
		x = x.reshape(n, self.h, self.w, self.k)
		x = util.im2patch(x, self.fh, self.fw, self.stride, self.pad, self.padding_mode, **self.padding_args).reshape(-1, self.k * self.fh * self.fw)
		self.x = x
		y = np.dot(x, self.W) + self.b # y : (n, h, w) * m
		return y.reshape(n, -1)

	# dy : n * (oh, ow, m)
	# ret : n * (h, w, k)
	def backward(self, dy):
		n = dy.shape[0]
		dy = dy.reshape(n, self.oh, self.ow, self.m).reshape(-1, self.m)
		self.dW = np.dot(self.x.T, dy) / n
		# b is common to filter
		# average of n, oh, ow dir
		self.db = np.mean(dy, axis = 0)

		dx = np.dot(dy, self.W.T)
		dx = dx.reshape(n, self.oh, self.ow, self.k, self.fh, self.fw)
		dx = util.patch2im(dx, self.h, self.w, self.stride, self.pad)
		return dx.reshape(n, -1)

	def get_weight(self):
		return { "W" : self.W, "b" : self.b }

	def get_grad(self):
		return { "W" : self.dW, "b" : self.db }

	def update_weight(self, delta):
		self.W += delta["W"]
		self.b += delta["b"]

class max_pooling_layer(abstract_layer):
	'''
	shape : list of int (input height, input width, num of channels)
	filter_shape : list of int (filter height, filter width)
	stride : list of int (vertical, horizontal)
	pad : list of int (top, botom, left, right)
	'''
	def __init__(self, shape, filter_shape, stride, pad, padding_mode = "constant", **padding_args):
		super().__init__()
		self.h = shape[0]
		self.w = shape[1]
		self.k = shape[2]
		self.fh = filter_shape[0]
		self.fw = filter_shape[1]
		self.stride = stride
		self.pad = pad
		self.padding_mode = padding_mode
		self.padding_args = padding_args
		self.oh = (self.h + pad[0] + pad[1] - self.fh) // stride[0] + 1
		self.ow = (self.w + pad[2] + pad[3] - self.fw) // stride[1] + 1
		self.argmax = None

	# x : n * (h, w, k)
	# ret : n * (oh, ow, k)
	def forward(self, x):
		n = x.shape[0]
		x = x.reshape(n, self.h, self.w, self.k)
		x = util.im2patch(x, self.fh, self.fw, self.stride, self.pad, self.padding_mode, **self.padding_args).reshape(-1, self.fh * self.fw)
		self.argmax = np.argmax(x, axis = 1)
		x = np.max(x, axis = 1)
		return x.reshape(n, self.oh, self.ow, self.k).reshape(n, -1)

	# dy : n * (oh, ow, k)
	# ret : n * (h, w, k)
	def backward(self, dy):
		n = dy.shape[0]
		dy = dy.ravel()
		dx = np.zeros((n * self.oh * self.ow * self.k, self.fh * self.fw))
		dx[np.arange(dy.size), self.argmax] = dy
		dx = dx.reshape(n, self.oh, self.ow, self.k, self.fh, self.fw)
		dx = util.patch2im(dx, self.h, self.w, self.stride, self.pad)
		return dx.reshape(n, -1)

class average_pooling_layer(abstract_layer):
	'''
	shape : list of int (input height, input width, num of channels)
	filter_shape : list of int (filter height, filter width)
	stride : list of int (vertical, horizontal)
	pad : list of int (top, botom, left, right)
	'''
	def __init__(self, shape, filter_shape, stride, pad, padding_mode = "constant", **padding_args):
		super().__init__()
		self.h = shape[0]
		self.w = shape[1]
		self.k = shape[2]
		self.fh = filter_shape[0]
		self.fw = filter_shape[1]
		self.stride = stride
		self.pad = pad
		self.padding_mode = padding_mode
		self.padding_args = padding_args
		self.oh = (self.h + pad[0] + pad[1] - self.fh) // stride[0] + 1
		self.ow = (self.w + pad[2] + pad[3] - self.fw) // stride[1] + 1

	# x : n * (h, w, k)
	# ret : n * (oh, ow, k)
	def forward(self, x):
		n = x.shape[0]
		x = x.reshape(n, self.h, self.w, self.k)
		x = im2patch(x, self.fh, self.fw, self.stride, self.pad, self.padding_mode, **self.padding_args).reshape(-1, self.fh * self.fw)
		x = np.mean(x, axis = 1)
		return x.reshape(n, self.oh, self.ow, self.k).reshape(n, -1)

	# dy : n * (oh, ow, k)
	# ret : n * (h, w, k)
	def backward(self, dy):
		n = dy.shape[0]
		filter_size = self.fh * self.fw
		dx = np.tile(dy.reshape(-1, 1), filter_size) / filter_size
		dx = dx.reshape(n, self.oh, self.ow, self.k, self.fh, self.fw)
		dx = patch2im(dx, self.h, self.w, self.stride, self.pad)
		return dx.reshape(n, -1)

'''
if activation function deponds on only one variable
'''
class activation_layer(abstract_layer):
	def __init__(self, f, df):
		super().__init__()
		self.f = f
		self.df = df
		self.x = None

	def forward(self, x):
		self.x = x
		return self.f(x)

	def backward(self, dy):
		return dy * self.df(self.x)

'''
if activation function depends on all variable in x
y = f(x)
df : N x m -> N x m x m, N = batch size, m = dim(x) = dim(y), returns N jacobian matrices
dy_j / dx_i = df_ij
'''
class multivariable_activation_layer(activation_layer):
	def __init__(self, f, df):
		super().__init__(f, df)

	def backward(self, dy):
		return np.matmul(dy[:, np.newaxis, :], np.transpose(self.df(self.x), axes = [0, 2, 1])).reshape(dy.shape[0], -1)

class batch_normalization_layer(learning_layer):
	'''
	if gamma or beta is None, gamma : 1.0, beta : 0.0
	'''
	def __init__(self, gamma = None, beta = None, eps = 1e-7):
		self.gamma = gamma
		self.beta = beta
		self.eps = eps
		self.var = None
		self.hat_x = None
		self.dgamma = None
		self.dbeta = None

	def forward(self, x):
		if self.gamma is None:
			self.gamma = np.ones_like(x[0])
		if self.beta is None:
			self.beta = np.zeros_like(x[0])
		mean = np.mean(x, axis = 0, keepdims = True)
		self.var = np.var(x, axis = 0)
		self.hat_x = (x - mean) / np.sqrt(self.var.reshape(1, -1) + self.eps)
		y = self.gamma.reshape(1, -1) * self.hat_x + self.beta.reshape(1, -1)
		return y

	def backward(self, dy):
		n = dy.shape[0]
		self.dgamma = np.sum(self.hat_x * dy, axis = 0)
		self.dbeta = np.sum(dy, axis = 0)
		dx = (self.gamma / np.sqrt(self.var + self.eps)).reshape(1, -1) * \
			(dy - (self.dbeta.reshape(1, -1) + self.hat_x * self.dgamma.reshape(1, -1)) / n)
		return dx

	def get_weight(self):
		return { "gamma" : self.gamma, "beta" : self.beta }

	def get_grad(self):
		return { "gamma" : self.dgamma, "beta" : self.dbeta }

	def update_weight(self, delta):
		self.gamma += delta["gamma"]
		self.beta += delta["beta"]

class dropout_layer(abstract_layer):
	def __init__(self, dropout_ratio):
		super().__init__()
		self.dropout_ratio = dropout_ratio
		self.is_test = False
		self.mask = None

	def forward(self, x):
		if self.is_test:
			self.is_test = False
			return x

		scale = 1 / (1 - self.dropout_ratio)
		net = np.random.rand(*x.shape) >= self.dropout_ratio
		self.mask = net * scale
		return x * self.mask

	def backward(self, dy):
		return dy * self.mask

	def test(self):
		self.is_test = True

'''
last_layer is layer with error function
'''
class last_layer(abstract_layer):
	def __init__(self):
		super().__init__()
		self.d = None
		self.error = None

	def set_data(self, d):
		self.d = d

	def error(self):
		return self.error
		

class error_layer(last_layer):
	def __init__(self, f, df):
		super().__init__()
		self.f = f
		self.df = df
		self.y = None

	def forward(self, y):
		self.y = y
		self.error = np.mean(self.f(self.y, self.d))
		return y

	def backward(self, dy):
		return self.df(self.y, self.d)

'''
faster than using multivariable_activation_layer and error_layer
'''
class soft_max_and_cross_entropy_error_layer(last_layer):
	def __init__(self):
		super().__init__()
		self.y = None

	def forward(self, x):
		self.y = func.soft_max(x)
		self.error = np.mean(func.cross_entropy_error(self.y, self.d))
		return self.y

	def backward(self, dy):
		return self.y - self.d

class ident_and_mean_squared_error_layer(last_layer):
	def __init__(self):
		super().__init__()
		self.y = None

	def forward(self, x):
		self.y = func.ident(x)
		self.error = np.mean(func.mean_squared_error(self.y, self.d))
		return self.y

	def backward(self, dy):
		return self.y - self.d

