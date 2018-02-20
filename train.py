import numpy as np
import matplotlib.pyplot as plt
import pickle
from neitwork import trainer as tr
from neitwork import layer
from neitwork import function as func
from neitwork import optimizer as opt
from neitwork import util

def read_pickle(path):
	obj = None
	with open(path, mode = "rb") as f:
		obj = pickle.load(f)
	return obj

def main():
	x_train = read_pickle("../dataset/train-images-idx3-ubyte.pkl")
	d_train = read_pickle("../dataset/train-labels-idx1-ubyte.pkl")
	x_test = read_pickle("../dataset/t10k-images-idx3-ubyte.pkl")
	d_test = read_pickle("../dataset/t10k-labels-idx1-ubyte.pkl")

	each_iter_num = 100
	call_learn_num = 50
	train_num = x_train.shape[0]

	batch_size = 50
	each_epoch = batch_size * each_iter_num / train_num
	train_num = x_train.shape[0]
	learning_rate = 1e-3
	dropout_ratio = 0.5

	'''
	configurate network
	'''
	N = []
	N.append(util.make_convolution_with_randn(shape = (28, 28, 1, 32), filter_shape = (5, 5), stride = (1, 1), pad = (2, 2, 2, 2), std_dev = func.he_init_std_dev))
	N.append(layer.batch_normalization_layer())
	N.append(layer.activation_layer(func.ramp, func.d_ramp))
	N.append(layer.max_pooling_layer(shape = (28, 28, 32), filter_shape = (2, 2), stride = (2, 2), pad = (0, 0, 0, 0)))
	N.append(util.make_convolution_with_randn(shape = (14, 14, 32, 64), filter_shape = (5, 5), stride = (1, 1), pad = (2, 2, 2, 2), std_dev = func.he_init_std_dev))
	N.append(layer.batch_normalization_layer())
	N.append(layer.activation_layer(func.ramp, func.d_ramp))
	N.append(layer.max_pooling_layer(shape = (14, 14, 64), filter_shape = (2, 2), stride = (2, 2), pad = (0, 0, 0, 0)))
	N.append(util.make_linear_with_randn(7 * 7 * 64, 1024, std_dev = func.he_init_std_dev))
	N.append(layer.batch_normalization_layer())
	N.append(layer.activation_layer(func.ramp, func.d_ramp))
	N.append(layer.dropout_layer(dropout_ratio))
	N.append(util.make_linear_with_randn(1024, 10, std_dev = func.he_init_std_dev))
	N.append(layer.batch_normalization_layer())
	N.append(layer.soft_max_and_cross_entropy_error_layer()) # faster
	# N.append(layer.multivariable_activation_layer(func.soft_max, func.d_soft_max))
	# N.append(layer.error_layer(func.cross_entropy_error, func.d_cross_entropy_error))

	train_accuracy_list = []
	test_accuracy_list = []

	'''
	optimizer : Adam
	'''
	trainer = tr.trainer(N, opt.adam(stepsize = learning_rate))

	for i in range(call_learn_num):
		trainer.learn(x_train, d_train, each_iter_num, batch_size)
		train_accuracy_list.append(accuracy(trainer, x_train[:1000], d_train[:1000], batch_size))
		test_accuracy_list.append(accuracy(trainer, x_test[:1000], d_test[:1000], batch_size))
		print(each_epoch * (i + 1), train_accuracy_list[-1], test_accuracy_list[-1], flush = True)

	train_accuracy = accuracy(trainer, x_train, d_train, batch_size)
	test_accuracy = accuracy(trainer, x_test, d_test, batch_size)
	print(train_accuracy, test_accuracy)

	print("write transition of accuracy to : ")
	accuracy_list_path = input()
	print("save the graph as : ")
	graph_path = input()
	print("write last accuracy to : ")
	last_accuracy_path = input()
	print("save network as : ")
	network_path = input()

	with open(accuracy_list_path, mode = "w") as f:
		for i in range(call_learn_num):
			text = str(each_epoch * (i + 1)) + " " + str(train_accuracy_list[i]) + " " + str(test_accuracy_list[i]) + "\n"
			f.write(text)

	with open(last_accuracy_path, mode = "w") as f:
		f.write(str(train_accuracy) + " " + str(test_accuracy))

	with open(network_path, mode = "wb") as f:
		pickle.dump(N, f)

	epoch_num = each_iter_num * batch_size * call_learn_num / train_num
	x_axis = np.linspace(epoch_num / call_learn_num, epoch_num, call_learn_num)
	plt.ylim([0, 1])
	plt.plot(x_axis, np.array(train_accuracy_list), label = "train accuracy")
	plt.plot(x_axis, np.array(test_accuracy_list), label = "test accuracy")
	plt.legend()
	plt.xlabel("epochs")
	plt.ylabel("accuracy")
	plt.savefig(graph_path)

def accuracy(trainer, x, d, batch_size):
	n = x.shape[0]
	x = np.split(x, np.arange(batch_size, n, step = batch_size), axis = 0)
	d = np.split(d, np.arange(batch_size, n, step = batch_size), axis = 0)
	s = 0
	for i in range(len(x)):
		# print("accuracy calculating : " + str(i) + " / " + str(len(x)), flush = True)
		trainer.test()
		y = trainer.forward_all(x[i])
		y = np.argmax(y, axis = 1)
		d_batch = np.argmax(d[i], axis = 1)
		s += np.sum(y == d_batch)
	return s / n

if __name__ == "__main__":
	main()

