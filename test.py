import numpy as np
from neitwork import *
from neitwork import layer
import pickle
import sys
from PIL import Image

def main():
	N = None
	with open(sys.argv[1], mode = "rb") as f:
		N = pickle.load(f)

	path = sys.argv[2]
	im = np.array(Image.open(path).convert("L"))
	im = (255 - im) / 255
	im = im.reshape(1, 28 * 28)
	tr = trainer.trainer(N, None)
	Y = tr.forward_all(N, im)
	Y = Y.flatten()

	max_num = 100
	for i in range(10):
		print(str(i) + " : " + "{:.2%}".format(Y[i]) + " " + "#" * int(max_num * Y[i]))

if __name__ == "__main__":
	main()
