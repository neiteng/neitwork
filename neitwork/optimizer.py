import numpy as np
from abc import ABCMeta, abstractmethod
from . import layer

class abstract_optimizer(metaclass = ABCMeta):
	@abstractmethod
	def update(self, l, state, optimize_state):
		pass

class SGD(abstract_optimizer):
	def __init__(self, learning_rate = 1e-2):
		self.learning_rate = learning_rate

	def update(self, l, state, optimize_state):
		grad = l.get_grad()
		delta = {}
		for key, g in grad.items():
			delta[key] = -g * self.learning_rate
		l.update_weight(delta)

class momentum_SGD(abstract_optimizer):
	def __init__(self, learning_rate = 1e-2, momentum = 0.9):
		self.learning_rate = learning_rate
		self.momentum = momentum

	def update(self, l, state, optimize_state):
		grad = l.get_grad()
		delta = {}
		if not "v" in state:
			state["v"] = {}
			for key, g in grad.items():
				state["v"][key] = np.zeros_like(g)
		v = state["v"]
		for key, g in grad.items():
			v[key] = self.momentum * v[key] - self.learning_rate * g
			delta[key] = v[key]
		l.update_weight(delta)

class ada_grad(abstract_optimizer):
	def __init__(self, learning_rate = 1e-3, eps = 1e-7):
		self.learning_rate = learning_rate
		self.eps = eps

	def update(self, l, state, optimize_state):
		grad = l.get_grad()
		delta = {}
		if not "h" in state:
			state["h"] = {}
			for key, g in grad.items():
				state["h"][key] = np.zeros_like(g)
		h = state["h"]
		for key, g in grad.items():
			h[key] += g * g
			delta[key] = -self.learning_rate / np.sqrt(h[key] + self.eps) * g
		l.update_weight(delta)

class adam(abstract_optimizer):
	def __init__(self, stepsize = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
		self.stepsize = stepsize
		self.beta1 = beta1
		self.beta2 = beta2
		self.eps = eps

	def update(self, l, state, optimize_state):
		grad = l.get_grad()
		t = optimize_state["t"]
		if not "m" in state:
			state["m"] = {}
			for key, g in grad.items():
				state["m"][key] = np.zeros_like(g)
		if not "v" in state:
			state["v"] = {}
			for key, g in grad.items():
				state["v"][key] = np.zeros_like(g)
		m = state["m"]
		v = state["v"]

		delta = {}
		for key, g in grad.items():
			m[key] = self.beta1 * m[key] + (1 - self.beta1) * g
			v[key] = self.beta2 * v[key] + (1 - self.beta2) * g * g
			M = m[key] / (1 - self.beta1 ** t)
			V = v[key] / (1 - self.beta2 ** t)
			delta[key] = -self.stepsize * M / (np.sqrt(V) + self.eps)
		l.update_weight(delta)

class rmsprop(abstract_optimizer):
	def __init__(self, stepsize = 1e-2, alpha = 0.99, eps = 1e-8):
		self.stepsize = stepsize
		self.alpha = alpha
		self.eps = eps

	def update(self, l, state, optimize_state):
		grad = l.get_grad()
		if not "m" in state:
			state["m"] = {}
			for key, g in grad.items():
				state["m"][key] = np.zeros_like(g)

		m = state["m"]
		delta = {}
		for key, g in grad.items():
			m[key] = self.alpha * m[key] + (1 - self.alpha) * g * g
			delta[key] = -self.stepsize / (np.sqrt(m[key]) + self.eps) * g
		l.update_weight(delta)

class ada_delta(abstract_optimizer):
	def __init__(self, rho = 0.95, eps = 1e-6):
		self.rho = rho
		self.eps = eps

	def update(self, l, state, optimize_state):
		grad = l.get_grad()
		if not "ex2" in state:
			state["ex2"] = {}
			for key, g in grad.items():
				state["ex2"][key] = np.zeros_like(g)
		if not "eg2" in state:
			state["eg2"] = {}
			for key, g in grad.items():
				state["eg2"][key] = np.zeros_like(g)
		eg2 = state["eg2"]
		ex2 = state["ex2"]
		delta = {}
		for key, g in grad.items():
			eg2[key] = self.rho * eg2[key] + (1 - self.rho) * g * g
			delta[key] = -np.sqrt(ex2[key] + self.eps) / np.sqrt(eg2[key] + self.eps) * g
			ex2[key] = self.rho * ex2[key] + (1 - self.rho) * (delta[key] ** 2)
		l.update_weight(delta)

