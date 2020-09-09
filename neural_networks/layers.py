"""
Author: Sophia Sanborn, Sagnik Bhattacharya
Institution: UC Berkeley
Date: Spring 2020
Course: CS189/289A
Website: github.com/sophiaas, github.com/sagnibak
"""

import numpy as np
from abc import ABC, abstractmethod

from neural_networks.activations import initialize_activation
from neural_networks.weights import initialize_weights
from neural_networks.utils.convolution import im2col, col2im, pad2d

from collections import OrderedDict

from typing import Callable, List, Tuple


class Layer(ABC):
    """Abstract class defining the `Layer` interface."""

    def __init__(self):
        self.activation = None

        self.n_in = None
        self.n_out = None

        self.parameters = {}
        self.cache = {}
        self.gradients = {}

        super().__init__()

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        pass

    def clear_gradients(self) -> None:
        self.cache = OrderedDict({a: [] for a, b in self.cache.items()})
        self.gradients = OrderedDict(
            {a: np.zeros_like(b) for a, b in self.gradients.items()}
        )

    def forward_with_param(
        self, param_name: str, X: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Call the `forward` method but with `param_name` as the variable with
        value `param_val`, and keep `X` fixed.
        """

        def inner_forward(param_val: np.ndarray) -> np.ndarray:
            self.parameters[param_name] = param_val
            return self.forward(X)

        return inner_forward

    def _get_parameters(self) -> List[np.ndarray]:
        return [b for a, b in self.parameters.items()]

    def _get_cache(self) -> List[np.ndarray]:
        return [b for a, b in self.cache.items()]

    def _get_gradients(self) -> List[np.ndarray]:
        return [b for a, b in self.gradients.items()]


def initialize_layer(
    name: str,
    activation: str = None,
    weight_init: str = None,
    n_out: int = None,
    kernel_shape: Tuple[int] = None,
    stride: int = None,
    pad: int = None,
    mode: str = None,
    keep_dim: str = "first",
) -> Layer:
    """Factory function for layers."""
    if name == "fully_connected":
        return FullyConnected(
            n_out=n_out, activation=activation, weight_init=weight_init,
        )

    elif name == "elman":
        return Elman(n_out=n_out, activation=activation, weight_init=weight_init,)

    elif name == "conv2d":
        return Conv2D(
            n_out=n_out,
            activation=activation,
            kernel_shape=kernel_shape,
            stride=stride,
            pad=pad,
            weight_init=weight_init,
        )

    else:
        raise NotImplementedError("Layer type {} is not implemented".format(name))


class FullyConnected(Layer):
    """A fully-connected layer multiplies its input by a weight matrix, adds
    a bias, and then applies an activation function.
    """

    def __init__(
        self, n_out: int, activation: str, weight_init="xavier_uniform"
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)

        # instantiate the weight initializer
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int]) -> None:
        """Initialize all layer parameters (weights, biases)."""
        self.n_in = X_shape[1]

        ### BEGIN YOUR CODE ###

        W = self.init_weights.__call__((self.n_in,self.n_out))
        b = np.zeros(self.n_out)



        self.parameters = OrderedDict({"W": W, "b": b})
        self.cache: OrderedDict = OrderedDict({"X": None, "Z": None})# what do you need cache for backprop?
        self.gradients: OrderedDict = OrderedDict({"W": np.zeros(W.shape), "b": np.zeros(b.shape)})
                                      # MUST HAVE SAME KEYS AS `self.parameters`

        ### END YOUR CODE ###

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: multiply by a weight matrix, add a bias, apply activation.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.

        Parameters
        ----------
        X  input matrix of shape (batch_size, input_dim)

        Returns
        -------
        a matrix of shape (batch_size, output_dim)
        """
        # initialize layer parameters if they have not been initialized
        if self.n_in is None:
            self._init_parameters(X.shape)

        ### BEGIN YOUR CODE ###

        # perform an affine transformation and activation
        v1=X@self.parameters["W"]
        Z=v1+self.parameters["b"]#this will add row-wise as is automatic in numpy
        o=self.activation.forward(Z)# should operate rowwise


        # store information necessary for backprop in `self.cache`
        self.cache["X"]=X
        self.cache["Z"]=Z
        ### END YOUR CODE ###

        return o

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for fully connected layer.
        Compute the gradients of the loss with respect to:
            1. the weights of this layer (mutate the `gradients` dictionary)
            2. the bias of this layer (mutate the `gradients` dictionary)
            3. the input of this layer (return this)

        Parameters
        ----------
        dLdY  derivative of the loss with respect to the output of this layer
              shape (batch_size, output_dim)

        Returns
        -------
        derivative of the loss with respect to the input of this layer
        shape (batch_size, input_dim)
        """
        ### BEGIN YOUR CODE ###

        # unpack the cache
        Z=self.cache["Z"]
        X=self.cache["X"]
        # compute the gradients of the loss w.r.t. all parameters as well as the
        # input of the layer

        dZ=self.activation.backward(Z,dLdY)
        dB=np.sum(dZ,axis=0)
        dW=X.T@dZ
        dX=dZ@(self.parameters["W"].T)
        # store the gradients in `self.gradients`
        # the gradient for self.parameters["W"] should be stored in
        # self.gradients["W"], etc.

        self.gradients["W"]=dW
        self.gradients["b"]=dB
        ### END YOUR CODE ###

        return dX


class Elman(Layer):
    """Elman recurrent layer."""

    def __init__(
        self,
        n_out: int,
        activation: str = "tanh",
        weight_init: str = "xavier_uniform",
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int]) -> None:
        """Initialize all layer parameters."""
        self.n_in = X_shape[1]

        ### BEGIN YOUR CODE ###

        W = self.init_weights.__call__((self.n_in,self.n_out))# initialize weights using self.init_weights
        U = self.init_weights.__call__((self.n_out,self.n_out))# initialize weights using self.init_weights
        b = np.zeros((1,self.n_out))# initialize biases to zeros



        # initialize the cache, save the parameters, initialize gradients
        self.parameters: OrderedDict = OrderedDict({"W": W, "U": U, "b": b})
        self.gradients: OrderedDict = OrderedDict({"W": np.zeros(W.shape),"U": np.zeros(U.shape), "b": np.zeros(b.shape)})
        #self.cache: OrderedDict = OrderedDict({"X": None, "Z": None})# what do you need cache for backprop?
        self.cache: OrderedDict = self._init_cache# what do you need cache for backprop?


        ### END YOUR CODE ###

    def _init_cache(self, X_shape: Tuple[int]) -> None:
        """Initialize the layer cache. This contains useful information for
        backprop, crucially containing the hidden states.
        """
        ### BEGIN YOUR CODE ###

        s0 = np.zeros((X_shape[0],self.n_out))# the first hidden state
        self.cache = OrderedDict({"s": [s0],"X": [], "Z": []})  # THIS IS INCOMPLETE

        ### END YOUR CODE ###

    def forward_step(self, X: np.ndarray) -> np.ndarray:
        """Compute a single recurrent forward step.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.

        `self.cache["s"]` is a list storing all previous hidden states.
        The forward step is computed as:
            s_t+1 = fn(W X + U s_t + b)

        Parameters
        ----------
        X  input matrix of shape (batch_size, input_dim)

        Returns
        -------
        a matrix of shape (batch_size, output_dim)
        """
        ### BEGIN YOUR CODE ###

        # perform a recurrent forward step
        self.cache["Z"].append(
            X@self.parameters["W"]+self.cache["s"][-1]@self.parameters["U"]+self.parameters["b"]
            )


        out=(self.activation.forward(self.cache["Z"][-1]))


        # store information necessary for backprop in `self.cache`

        ### END YOUR CODE ###

        return out
        #gets appended to self.cache["s"] right after.

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the forward pass for `t` time steps. This should involve using
        forward_step repeatedly, possibly in a loop. This should be fairly simple
        since `forward_step` is doing most of the heavy lifting.

        Parameters
        ----------
        X  input matrix containing inputs for `t` time steps
           shape (batch_size, input_dim, t)

        Returns
        -------
        the final output/hidden state
        shape (batch_size, output_dim)
        """
        if self.n_in is None:
            self._init_parameters(X.shape[:2])


        self._init_cache(X.shape)

        ### BEGIN YOUR CODE ###
        self.cache["X"] = np.copy(X)

        # perform `t` forward passes through time and return the last
        # hidden/output state
        for t in range(X.shape[2]):
            self.cache["s"].append(self.forward_step(X[:,:,t]))

        ### END YOUR CODE ###

        return self.cache["s"][-1]

    def backward(self, dLdY: np.ndarray) -> List[np.ndarray]:
        """Backward pass for recurrent layer. Compute the gradient for all the
        layer parameters as well as every input at every time step.

        Parameters
        ----------
        dLdY  derivative of loss with respect to output of this layer
              shape (batch_size, output_dim)

        Returns
        -------
        list of numpy arrays of shape (batch_size, input_dim) of length `t`
        containing the derivative of the loss with respect to the input at each
        time step
        """
        ### BEGIN YOUR CODE ###


        '''
        Think of timesteps as:
        -1: corresponding to fake starting point- no corresponding xt
        0,1,2,3,...,T: we use save this state to feed back into network - has corresponding xt
        The state s[T] is the output of the layers

        self.cache["s"][i+1] contains s_{i}
        self.cache["s"][0] contains the starting s0
        '''

        T=len(self.cache["s"])-2#although this includes an initialisation s
        assert T==self.cache["X"].shape[2]-1

        dLdX=np.zeros(self.cache["X"].shape)


        dzts_dict={}
        dsts_dict={T:dLdY}

        for t in range(T,-1,-1):#first element fake so don`t include tb=T
            #cycle through t=T-1,T-2,...,1
            assert len(self.cache["Z"])==T+1
            dzts_dict[t]=self.activation.backward(self.cache["Z"][t],dsts_dict[t])
            dLdX[:,:,t]=dzts_dict[t]@(self.parameters["W"].T)
            if t>=1:#don`t need to run this on the last iteration.
                dsts_dict[t-1]=dzts_dict[t]@(self.parameters["U"].T)

        t=T
        dB=np.sum(dzts_dict[t],axis=0)
        dW=np.matmul(self.cache["X"][:,:,t].T,dzts_dict[t])
        dU=np.matmul(self.cache["s"][t].T,dzts_dict[t])#self.cache["s"][t] refers to s_(t-1) for t>=0 else it referes to the initialisation of s.

        for t in range(T-1,-1,-1):#first element fake so don`t include tb=T
            #cycle through t=T-1,T-2,...,1
            #could optimize this with tensordot
            dB+=np.sum(dzts_dict[t],axis=0)
            dW+=(self.cache["X"][:,:,t].T)@dzts_dict[t]
            dU+=(self.cache["s"][t].T)@dzts_dict[t]#self.cache["s"][t] refers to s_(t-1)) for t>=0 else it referes to the initialisation of s.


        # perform backpropagation through time, storing the gradient of the loss
        # w.r.t. each time step in `dLdX`

        ### END YOUR CODE ###


        self.gradients["W"]=dW
        self.gradients["U"]=dU
        self.gradients["b"]=dB

        #assert False
        return dLdX


    '''below are test methods'''
    def forward_step_sol(self, X: np.ndarray) -> np.ndarray:
        """Compute a single recurrent forward step.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.
        `self.cache["s"]` is a list storing all previous hidden states.
        The forward step is computed as:
        s_t+1 = fn(W X + U s_t + b)
        """
        W = self.parameters["W"]
        U = self.parameters["U"]
        b = self.parameters["b"]
        s = self.cache["s"]
        Z = (X @ W) + (s[-1] @ U) + b
        out = self.activation(Z)
        self.cache["Z"].append(Z)
        self.cache["s"].append(out)
        self.cache["X"].append(X)
        return out

    def forward_sol(self, X: np.ndarray) -> np.ndarray:
        """Compute the forward pass for `t` time steps. This should involve using
        forward_step repeatedly, possibly in a loop. This should be fairly simple
        since `forward_step` is doing most of the heavy lifting.
        """
        if self.n_in is None:
            self._init_parameters(X.shape[:2])
        self._init_cache(X.shape)
        Y = []
        for t in range(X.shape[2]):
            y = self.forward_step_sol(X[:, :, t])
            Y.append(y)
        return Y[-1]

    def backward_step_sol(
        self, dLdYt: np.ndarray, t: int
        ) -> Tuple[np.ndarray, np.ndarray]:
        """Take one backward step. Compute the gradient of the loss with respect
        to all the parameters and update them. Return the gradient of the loss
        with respect to the previous hidden state and the current input.
        Parameters
        ----------
        dLdYt gradient of the loss w.r.t. the output at time step t
        t the current time step
        Returns
        -------
        a tuple containing
        - the gradient of the loss w.r.t. the previous hidden state
        - the gradient of the loss w.r.t. the current input
        """
        W = self.parameters["W"]
        U = self.parameters["U"]
        Xt = self.cache["X"][t] # current X
        Zt = self.cache["Z"][t] # current h
        Yt = self.cache["s"][t] # previous s
        dZt = self.activation.backward(Zt, dLdYt)
        dXt = dZt @ W.T
        dLdYt_prev = dZt @ U.T
        self.gradients["W"] += Xt.T @ dZt
        self.gradients["U"] += Yt.T @ dZt
        self.gradients["b"] += dZt.sum(axis=0, keepdims=True)
        return dLdYt_prev, dXt


    def backward_sol(self, dLdY: np.ndarray) -> List[np.ndarray]:
        """Backward pass for recurrent layer. Compute the gradient for all the
        layer parameters as well as every input at every time step. This should
        be a fairly simple wrapper around `backward_step`.
        Parameters
        ----------
        dLdY derivative of loss with respect to output of this layer
        shape (batch_size, output_dim)
        Returns
        -------
        list of numpy arrays of shape (batch_size, input_dim) of length `t`
        containing the derivative of the loss with respect to the input at each
        time step
        """
        if self.gradients["b"].shape[0]!=1:
            self.gradients["b"]=self.gradients["b"].reshape((1,self.gradients["b"].shape[0]))
        dLdX = []
        dLdYt = dLdY
        for t in reversed(range(len(self.cache["X"]))):
            dLdYt, dXt = self.backward_step_sol(dLdYt, t)
            dLdX.insert(0, dXt)
        return dLdX


class Conv2D(Layer):
    """Convolutional layer for inputs with 2 spatial dimensions."""

    def __init__(
        self,
        n_out: int,
        kernel_shape: Tuple[int],
        activation: str,
        stride: int = 1,
        pad: str = "same",
        weight_init: str = "xavier_uniform",
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.pad = pad

        self.activation = initialize_activation(activation)
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int]) -> None:
        """Initialize all layer parameters."""
        ### BEGIN YOUR CODE ###

        # initialize weights, biases, the cache, and gradients

        self.n_in=X_shape[3]

        W = self.init_weights.__call__((self.kernel_shape[0],self.kernel_shape[1],self.n_in,self.n_out))# initialize weights using self.init_weights
        b = self.init_weights.__call__((1,self.n_out))# initialize biases to zeros

        self.parameters: OrderedDict = OrderedDict({"W": W, "b": b})
        self.cache: OrderedDict = OrderedDict({"X": None, "Z": None})
        self.gradients: OrderedDict = OrderedDict({"W": np.zeros(W.shape), "b": np.zeros(b.shape)})



        ### END YOUR CODE ###

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for convolutional layer. This layer convolves the input
        `X` with a filter of weights, adds a bias term, and applies an activation
        function to compute the output. This layer also supports padding and
        integer strides. Intermediates necessary for the backward pass are stored
        in the cache.

        Parameters
        ----------
        X  input with shape (batch_size, in_rows, in_cols, in_channels)

        Returns
        -------
        output feature maps with shape (batch_size, out_rows, out_cols, out_channels)
        """
        if self.n_in is None:
            self._init_parameters(X.shape)

        self.cache["orX"]=np.copy(X)
        W = self.parameters["W"]
        b = self.parameters["b"]

        kernel_height, kernel_width, in_channels, out_channels = W.shape
        kernel_shape = (kernel_height, kernel_width)

        stride=self.stride
        if self.pad=="same":

            #formula for padding with stride looked up online
            #https://www.pico.net/kb/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-tensorflow
            self.out_height=int(np.ceil((X.shape[1])/stride))
            self.out_width=int(np.ceil((X.shape[2])/stride))

            self.pad_along_height=np.max([0,(self.out_height-1)*stride+kernel_height-X.shape[1]])
            self.pad_along_width=np.max([0,(self.out_width-1)*stride+kernel_width-X.shape[2]])

            self.pad_top=self.pad_along_height//2
            self.pad_left=self.pad_along_width//2
            self.pad_bottom=self.pad_along_height-self.pad_top
            self.pad_right=self.pad_along_width-self.pad_left

            X=np.pad(X,[(0,0),
             (self.pad_top,self.pad_bottom),(self.pad_left,self.pad_right),(0,0) ])
        else:#valid
            self.out_height=int(np.ceil((X.shape[1]-kernel_height+1)/stride))
            self.out_width=int(np.ceil((X.shape[2]-kernel_width+1)/stride))
            self.pad_along_height=0
            self.pad_along_width=0

            #all the below is 0
            self.pad_top=self.pad_along_height//2
            self.pad_left=self.pad_along_width//2
            self.pad_bottom=self.pad_along_height-self.pad_top
            self.pad_right=self.pad_along_width-self.pad_bottom

        n_examples, in_rows, in_cols, in_channels = X.shape

        ### BEGIN YOUR CODE ###

        # implement a convolutional forward pass

        Z=np.zeros((X.shape[0],self.out_height,self.out_width,out_channels))

        # cache any values required for backprop

        W=self.parameters["W"]
        b=self.parameters["b"]
        for d1 in range(Z.shape[1]):
            for d2 in range(Z.shape[2]):
                for n in range(Z.shape[3]):
                    Z[:,d1,d2,n]=np.sum(
                        W[:,:,:,n]*
                        X[:#this axis is maintained
                        ,d1*stride:(d1*stride+W.shape[0]),d2*stride:(d2*stride+W.shape[1]),:],axis=(1,2,3))+b[0,n]#just don`t sum the batches together



        out=self.activation.forward(Z)

        self.cache["X"]=X
        self.cache["Z"]=Z
        ### END YOUR CODE ###
        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for conv layer. Computes the gradients of the output
        with respect to the input feature maps as well as the filter weights and
        biases.

        Parameters
        ----------
        dLdY  derivative of loss with respect to output of this layer
              shape (batch_size, out_rows, out_cols, out_channels)

        Returns
        -------
        derivative of the loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, in_channels)
        """
        ### BEGIN YOUR CODE ###

        # perform a backward pass
        X=self.cache["X"]
        Z=self.cache["Z"]
        W=self.parameters["W"]
        stride=self.stride

        dZ=self.activation.backward(Z,dLdY)

        dB=np.sum(dZ,(0,1,2))#don`t sum over the last dimension

        dW=np.zeros(W.shape)

        for i3 in range(W.shape[0]):
            for i4 in range(W.shape[1]):
                for i5 in range(W.shape[2]):
                    for i6 in range(W.shape[3]):

                        dW[i3,i4,i5,i6]+=np.sum(np.tensordot(
                            (X[:,i3:(i3+Z.shape[1]*stride):stride,i4:(i4+Z.shape[2]*stride):stride,i5])[:,:dZ.shape[1],:dZ.shape[2]],
                            dZ[:,:,:,i6],axes=[[0,1,2],[0,1,2]]
                            ))

        orX=self.cache["orX"]
        dX=np.zeros(X.shape)#(batch_size, in_rows, in_cols, in_channels)



        for i0 in range(dZ.shape[1]):
            for i1 in range(dZ.shape[2]):

                dX[:,i0*stride:i0*stride+dW.shape[0],i1*stride:i1*stride+dW.shape[1],:]+=np.tensordot(dZ[:,i0,i1,:],(W[::,::,:,:]),[[1],[3]])

                ##          *                 *
                #dZ[:,i0,i1,:]*(W[::-1,::-1,:,:])

        self.gradients["W"]=dW
        self.gradients["b"]=dB

        ### END YOUR CODE ###

        if self.pad=="same":
            return dX[:,self.pad_top:-self.pad_bottom,self.pad_left:-self.pad_right,:]

        return dX

    def forward_faster(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for convolutional layer. This layer convolves the input
        `X` with a filter of weights, adds a bias term, and applies an activation
        function to compute the output. This layer also supports padding and
        integer strides. Intermediates necessary for the backward pass are stored
        in the cache.

        This implementation uses `im2col` which allows us to use fast general
        matrix multiply (GEMM) routines implemented by numpy. This is still
        rather slow compared to GPU acceleration, but still LEAGUES faster than
        the nested loop in the naive implementation.

        DO NOT ALTER THIS METHOD.

        You will write your naive implementation in forward().
        We will use forward_faster() to check your method.

        Parameters
        ----------
        X  input with shape (batch_size, in_rows, in_cols, in_channels)

        Returns
        -------
        output feature maps with shape (batch_size, out_rows, out_cols, out_channels)
        """
        if self.n_in is None:
            self._init_parameters(X.shape)

        W = self.parameters["W"]
        b = self.parameters["b"]

        kernel_height, kernel_width, in_channels, out_channels = W.shape
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_shape = (kernel_height, kernel_width)

        X_col, p = im2col(X, kernel_shape, self.stride, self.pad)
        out_rows = int((in_rows + p[0] + p[1] - kernel_height) / self.stride + 1)
        out_cols = int((in_cols + p[2] + p[3] - kernel_width) / self.stride + 1)

        W_col = W.transpose(3, 2, 0, 1).reshape(out_channels, -1)

        Z = (
            (W_col @ X_col)
            .reshape(out_channels, out_rows, out_cols, n_examples)
            .transpose(3, 1, 2, 0)
        )
        Z += b
        out = self.activation(Z)

        self.cache["Z"] = Z
        self.cache["X"] = X



        return out

    def backward_faster(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for conv layer. Computes the gradients of the output
        with respect to the input feature maps as well as the filter weights and
        biases.

        This uses im2col, so it is considerably faster than the naive implementation
        even on a CPU.

        DO NOT ALTER THIS METHOD.

        You will write your naive implementation in backward().
        We will use backward_faster() to check your method.

        Parameters
        ----------
        dLdY  derivative of loss with respect to output of this layer
              shape (batch_size, out_rows, out_cols, out_channels)

        Returns
        -------
        derivative of the loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, in_channels)
        """
        W = self.parameters["W"]
        b = self.parameters["b"]
        Z = self.cache["Z"]
        X = self.cache["X"]

        kernel_height, kernel_width, in_channels, out_channels = W.shape
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_shape = (kernel_height, kernel_width)

        dZ = self.activation.backward(Z, dLdY)

        dZ_col = dZ.transpose(3, 1, 2, 0).reshape(dLdY.shape[-1], -1)
        X_col, p = im2col(X, kernel_shape, self.stride, self.pad)
        W_col = W.transpose(3, 2, 0, 1).reshape(out_channels, -1).T

        dW = (
            (dZ_col @ X_col.T)
            .reshape(out_channels, in_channels, kernel_height, kernel_width)
            .transpose(2, 3, 1, 0)
        )
        dB = dZ_col.sum(axis=1).reshape(1, -1)

        dX_col = W_col @ dZ_col
        dX = col2im(dX_col, X, W.shape, self.stride, p).transpose(0, 2, 3, 1)

        self.gradients["W"] = dW
        self.gradients["b"] = dB

        return dX
