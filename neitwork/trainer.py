import numpy as np
from . import layer

# network is a list of layers
# the last layer should be a subclass of layer.last_layer

class trainer:
	def __init__(self, network, optimizer, weight_decay = 0):
		self.network = network
		self.optimizer = optimizer
		self.state = [{ "self" : l } for l in network]
		self.optimize_state = { "t" : 0 }
		self.weight_decay = weight_decay

	def forward_all(self, x):
		for l in self.network:
			x = l.forward(x)
		return x

	def backward_all(self):
		dy = 1
		# back propagation
		for l in reversed(self.network):
			dy = l.backward(dy)

			# weigt decay
			if issubclass(type(l), layer.learning_layer):
				grad = l.get_grad()
				weight = l.get_weight()
				grad["W"] += self.weight_decay * weight["W"]

	def set_data(self, d):
		self.network[-1].set_data(d)

	def learn(self, x, d, iter_num, batch_size):
		N = x.shape[0]

		for i in range(iter_num):
			self.optimize_state["t"] += 1
			# print("iterating : " + str(i)  + " / " + str(iter_num), flush = True)
			batch_index = np.random.choice(N, batch_size)
			x_batch = x[batch_index]
			d_batch = d[batch_index]

			self.set_data(d_batch)
			self.forward_all(x_batch)
			self.backward_all()

			for i, l in enumerate(self.network):
				if issubclass(type(l), layer.learning_layer):
					self.optimizer.update(l, self.state[i], self.optimize_state)

	def test(self):
		for l in self.network:
			if issubclass(type(l), layer.dropout_layer):
				l.test()

