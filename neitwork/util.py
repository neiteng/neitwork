from . import layer
import numpy as np

'''
m : input num, n : output num
'''
def make_linear_with_randn(m, n, std_dev):
	return layer.linear_layer(np.random.randn(m, n) * std_dev(m), np.zeros(n))

def make_convolution_with_randn(shape, filter_shape, stride, pad, std_dev, padding_mode = "constant"):
	return layer.convolution_layer(np.random.randn(shape[2] * filter_shape[0] * filter_shape[1], shape[3]) * std_dev(shape[0] * shape[1] * shape[2]), np.zeros(shape[3]), shape, filter_shape, stride, pad, padding_mode)

'''
x : n * h * w * k
out : n * oh * ow * k * fh * fw
'''
def im2patch(x, fh, fw, stride, pad, padding_mode, **padding_args):
	n, h, w, k = x.shape
	oh = (h + pad[0] + pad[1] - fh) // stride[0] + 1
	ow = (w + pad[2] + pad[3] - fw) // stride[1] + 1
	# zero padding
	x = x.transpose(1, 2, 0, 3)
	x = np.pad(x, [(pad[0], pad[1]), (pad[2], pad[3]), (0, 0), (0, 0)], mode = padding_mode, **padding_args)
	# now, x : h, w, n, k
	# out : fh, fw, oh, ow, n, k
	out = np.empty((fh, fw, oh, ow, n, k))

	for top in range(fh):
		bottom = top + oh * stride[0]
		for left in range(fw):
			right = left + ow * stride[1]
			out[top, left, :, :, :, :] = x[top : bottom : stride[0], left : right : stride[1], :, :]

	# n, oh, ow, k, fh, fw
	out = out.transpose(4, 2, 3, 5, 0, 1)
	return out

'''
x : n * oh * ow * k * fh * fw
out : n * h * w * k
'''
def patch2im(x, h, w, stride, pad):
	n, oh, ow, k, fh, fw = x.shape
	x = x.transpose(4, 5, 1, 2, 0, 3)
	# now, fh, fw, oh, ow, n, k
	# out : h, w, n, k
	out = np.zeros((fh + oh * stride[0], fw + ow * stride[1], n, k))
	for top in range(fh):
		bottom = top + stride[0] * oh
		for left in range(fw):
			right = left + stride[1] * ow
			out[top : bottom : stride[0], left : right : stride[1], :, :] += x[top, left, :, :, :, :]

	out = out[pad[0] : h + pad[0], pad[2] : w + pad[2], :, :]
	out = out.transpose(2, 0, 1, 3)
	return out # n * h * w * k
