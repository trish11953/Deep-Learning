from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class sequential(object):
    def __init__(self, *args):
        """
        Sequential Object to serialize the NN layers
        Please read this code block and understand how it works
        """
        self.params = {}
        self.grads = {}
        self.layers = []
        self.paramName2Indices = {}
        self.layer_names = {}

        # process the parameters layer by layer
        for layer_cnt, layer in enumerate(args):
            for n, v in layer.params.items():
                self.params[n] = v
                self.paramName2Indices[n] = layer_cnt
            for n, v in layer.grads.items():
                self.grads[n] = v
            if layer.name in self.layer_names:
                raise ValueError("Existing name {}!".format(layer.name))
            self.layer_names[layer.name] = True
            self.layers.append(layer)

    def assign(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].params[name] = val

    def assign_grads(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].grads[name] = val

    def get_params(self, name):
        # return the parameters by name
        return self.params[name]

    def get_grads(self, name):
        # return the gradients by name
        return self.grads[name]

    def gather_params(self):
        """
        Collect the parameters of every submodules
        """
        for layer in self.layers:
            for n, v in layer.params.items():
                self.params[n] = v

    def gather_grads(self):
        """
        Collect the gradients of every submodules
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] = v

    def apply_l1_regularization(self, lam):
        """
        Gather gradients for L1 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                param = self.params[n]
                grad = (param > 0).astype(np.float32) - (param < 0).astype(np.float32)
                self.grads[n] += lam * grad

    def apply_l2_regularization(self, lam):
        """
        Gather gradients for L2 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] += lam * self.params[n]

    def load(self, pretrained):
        """
        Load a pretrained model by names
        """
        for layer in self.layers:
            if not hasattr(layer, "params"):
                continue
            for n, v in layer.params.items():
                if n in pretrained.keys():
                    layer.params[n] = pretrained[n].copy()
                    print("Loading Params: {} Shape: {}".format(n, layer.params[n].shape))


class ConvLayer2D(object):
    def __init__(self, input_channels, kernel_size, number_filters,
                 stride=1, padding=0, init_scale=.02, name="conv"):

        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.number_filters = number_filters
        self.stride = stride
        self.padding = padding

        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(kernel_size, kernel_size,
                                                                input_channels, number_filters)
        self.params[self.b_name] = np.zeros(number_filters)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None

    def get_output_size(self, input_size):
        '''
        :param input_size - 4-D shape of input image tensor (batch_size, in_height, in_width, in_channels)
        :output a 4-D shape of the output after passing through this layer (batch_size, out_height, out_width, out_channels)
        '''
        output_shape = [None, None, None, None]
        #############################################################################
        # TODO: Implement the calculation to find the output size given the         #
        # parameters of this convolutional layer.                                   #
        #############################################################################
        batch_size, in_height, in_width, in_channels = input_size

        # Calculate output height and width
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        output_shape = (batch_size, out_height, out_width, self.number_filters)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output_shape

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        output_shape = self.get_output_size(img.shape)
        _, input_height, input_width, _ = img.shape
        _, output_height, output_width, _ = output_shape

        #############################################################################
        # TODO: Implement the forward pass of a single convolutional layer.       #
        # Store the results in the variable "output" provided above.                #
        #############################################################################
        if self.padding > 0:
            img = np.pad(img, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant',
                         constant_values=0)
        output = np.zeros(output_shape)
        for h in range(0, output_height):
            for w in range(0, output_width):
                ih, iw, weights, bias = h * self.stride, w * self.stride, self.params[self.w_name], self.params[
                    self.b_name]
                sliced = img[:, ih: ih + self.kernel_size, iw: iw + self.kernel_size, :]
                sliced = np.expand_dims(sliced, axis=-1)
                output[:, h, w, :] = np.sum(np.multiply(weights, sliced), axis=(1, 2, 3)) + bias
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = img
        return output

    def backward(self, dprev):
        img = self.meta
        if img is None:
            raise ValueError("No forward function called before for this module!")

        dimg, self.grads[self.w_name], self.grads[self.b_name] = None, None, None

        #############################################################################
        # TODO: Implement the backward pass of a single convolutional layer.        #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################

        _, out_h, out_w, _ = dprev.shape
        self.grads[self.w_name] = np.zeros_like(self.params[self.w_name])
        self.grads[self.b_name] = np.zeros_like(self.params[self.b_name])

        dpad, batch_size, num_fils = np.zeros_like(img), img.shape[0], self.number_filters

        for i in range(out_h):
            for j in range(out_w):
                h_s, w_s = i * self.stride, j * self.stride
                h_e, w_e = h_s + self.kernel_size, w_s + self.kernel_size

                sliced = img[:, h_s:h_e, w_s:w_e, :].reshape(batch_size, self.kernel_size, self.kernel_size,
                                                             img.shape[-1], 1)
                reshaped = dprev[:, i, j, :].reshape(batch_size, 1, 1, 1, num_fils)
                dpad[:, h_s:h_e, w_s:w_e, :] += np.sum(self.params[self.w_name] * reshaped, axis=4)
                self.grads[self.w_name] += np.sum(sliced * reshaped, axis=0)
                self.grads[self.b_name] += np.sum(reshaped, axis=(0, 1, 2, 3))

        if self.padding > 0:
            dimg = dpad[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            dimg = dpad

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        self.meta = None
        return dimg


class MaxPoolingLayer(object):
    def __init__(self, pool_size, stride, name):
        self.name = name
        self.pool_size = pool_size
        self.stride = stride
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        #############################################################################
        # TODO: Implement the forward pass of a single maxpooling layer.            #
        # Store your results in the variable "output" provided above.               #
        #############################################################################
        N, H, W, C = img.shape
        H_out = int(1 + (H - self.pool_size) / self.stride)
        W_out = int(1 + (W - self.pool_size) / self.stride)
        output = np.zeros((N, H_out, W_out, C))

        for i in range(N):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * self.stride
                    h_end = h_start + self.pool_size
                    w_start = w * self.stride
                    w_end = w_start + self.pool_size
                    window = img[i, h_start:h_end, w_start:w_end, :]
                    output[i, h, w, :] = np.amax(window, axis=(0, 1))

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = img
        return output

    def backward(self, dprev):
        img = self.meta

        dimg = np.zeros_like(img)
        batch_size, h_out, w_out, channels = dprev.shape
        h_pool, w_pool = self.pool_size, self.pool_size

        #############################################################################
        # TODO: Implement the backward pass of a single maxpool layer.              #
        # Store the computed gradients in self.grads with corresponding name.       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.pool_size
                w_end = w_start + self.pool_size
                window = img[:, h_start:h_end, w_start:w_end, :]  # N, k, k, C
                mask = window == np.max(window, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
                for b in range(batch_size):
                    for c in range(channels):
                        dimg[b, h_start:h_end, w_start:w_end, c] += mask[b, :, :, c] * dprev[b, i, j, c]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dimg
