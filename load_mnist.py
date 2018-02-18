import numpy as np
import pickle

def read_image(path):
	with open(path, mode = "rb") as f:
		src = f.read()
		num = int.from_bytes(src[4 : 8], "big")
		img_size = int.from_bytes(src[8 : 12], "big") * int.from_bytes(src[12 : 16], "big")
		img = [ x / 255.0 for x in src[16 : 16 + num * img_size]]
		return np.array(img).reshape(-1, img_size)

def read_label(path):
	with open(path, mode = "rb") as f:
		src = f.read()
		num = int.from_bytes(src[4 : 8], "big")
		L = np.zeros((num, 10))
		for i in range(num):
			L[i][src[8 + i]] = 1.0
		return L

def main():
	x_train_path = "../dataset/train-images-idx3-ubyte"
	d_train_path = "../dataset/train-labels-idx1-ubyte"
	x_test_path = "../dataset/t10k-images-idx3-ubyte"
	d_test_path = "../dataset/t10k-labels-idx1-ubyte"
	x_train = read_image(x_train_path)
	d_train = read_label(d_train_path)
	x_test = read_image(x_test_path)
	d_test = read_label(d_test_path)

	with open(x_train_path + ".pkl", mode = "wb") as f:
		pickle.dump(x_train, f)
	with open(d_train_path + ".pkl", mode = "wb") as f:
		pickle.dump(d_train, f)
	with open(x_test_path + ".pkl", mode = "wb") as f:
		pickle.dump(x_test, f)
	with open(d_test_path + ".pkl", mode = "wb") as f:
		pickle.dump(d_test, f)

if __name__ == "__main__":
	main()
