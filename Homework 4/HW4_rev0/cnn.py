import numpy as np

from layers import relu_forward, fc_forward, fc_backward, relu_backward, softmax_loss
from cnn_layers import conv_forward, conv_backward, max_pool_forward, max_pool_backward


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on the ipython notebook.
    """
    print("Hello from cnn.py!")


class ConvNet(object):
    """
    A convolutional network with the following architecture:

    conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - fc - relu - fc - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(1, 28, 28), num_filters_1=6, num_filters_2=16, filter_size=5,
               hidden_dim=100, num_classes=10, dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters_1: Number of filters to use in the first convolutional layer
        - num_filters_2: Number of filters to use in the second convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.dtype = dtype
        (self.C, self.H, self.W) = input_dim
        self.filter_size = filter_size
        self.num_filters_1 = num_filters_1
        self.num_filters_2 = num_filters_2
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Your initializations should work for any valid input dims,      #
        # number of filters, hidden dims, and num_classes. Assume that we use      #
        # max pooling with pool height and width 2 with stride 2.                  #
        #                                                                          #
        # For Linear layers, weights and biases should be initialized from a       #
        # uniform distribution from -sqrt(k) to sqrt(k),                           #
        # where k = 1 / (#input features)                                          #
        # For Conv. layers, weights should be initialized from a uniform           #
        # distribution from -sqrt(k) to sqrt(k),                                   #
        # where k = 1 / ((#input channels) * filter_size^2)                        #
        # Note: we use the same initialization as pytorch.                         #
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html           #
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html           #
        #                                                                          #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights for the convolutional layer using the keys 'W1' and 'W2'   #
        # (here we do not consider the bias term in the convolutional layer);      #
        # use keys 'W3' and 'b3' for the weights and biases of the                 #
        # hidden fully-connected layer, and keys 'W4' and 'b4' for the weights     #
        # and biases of the output affine layer.                                   #
        #                                                                          #
        # Make sure you have initialized W1, W2, W3, W4, b3, and b4 in the         #
        # params dicitionary.                                                      #
        #                                                                          #
        # Hint: The output of max pooling after W2 needs to be flattened before    #
        # it can be passed into W3. Calculate the size of W3 dynamically           #
        ############################################################################
        # conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - fc - relu - fc - softmax
        #After first conv layer
        conv1_out_size = ((self.H - filter_size)+1, (self.W-filter_size)+1)
        #After first pool layer
        pool1_out_size = (conv1_out_size[0]//2, conv1_out_size[1]// 2) # // for floor divison  
        #After second conv layer
        conv2_out_size = ((pool1_out_size[0] - filter_size)+1, (pool1_out_size[1]-filter_size)+1)  
        #After second pool layer
        pool2_out_size = (conv2_out_size[0] // 2, conv2_out_size[1] // 2)
        #Flatten size for the first fully connected layer
        flatten_size = num_filters_2 * pool2_out_size[0] * pool2_out_size[1]  
        
        #Initialize weight and bias
        #Conv layers
        weight_scale_conv1 = np.sqrt(1/(self.C * filter_size**2))
        weight_scale_conv2 = np.sqrt(1/(num_filters_1 * filter_size**2))
        #Fully connected layers
        weight_scale_fc1 = np.sqrt(1 / flatten_size)
        weight_scale_fc2 = np.sqrt(1 /  self.hidden_dim)

        W1 = np.random.uniform(-weight_scale_conv1, weight_scale_conv1, (num_filters_1,self.C,filter_size,filter_size))
        W2 = np.random.uniform(-weight_scale_conv2, weight_scale_conv2, (num_filters_2,num_filters_1, filter_size, filter_size))
        W3 = np.random.uniform(-weight_scale_fc1, weight_scale_fc1, (flatten_size, self.hidden_dim))
        b3 = np.zeros(self.hidden_dim)
        W4 = np.random.uniform(-weight_scale_fc2, weight_scale_fc2, (self.hidden_dim, self.num_classes))
        b4 = np.zeros(self.num_classes)


        self.params = {'W1': W1, 'W2': W2, 'W3': W3, 'b3': b3, 'W4': W4, 'b4': b4}



        
        # raise NotImplementedError("TODO: Add your implementation here.")
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1 = self.params['W1']
        W2 = self.params['W2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        # Hint: The output of max pooling after W2 needs to be flattened before    #
        # it can be passed into W3.                                                #
        ############################################################################
        # conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - fc - relu - fc - softmax       
        out1, cache1 = conv_forward(X, W1)
        out2, cache2 = relu_forward(out1)
        out3, cache3 = max_pool_forward(out2, pool_param)
        out4, cache4 = conv_forward(out3, W2)
        out5, cache5 = relu_forward(out4)
        out6, cache6 = max_pool_forward(out5, pool_param)
        
        N,F,H,W = np.shape(out6)
        out6_flat = out6.reshape((N,-1))

        out7, cache7 = fc_forward(out6_flat, W3, b3)
        out8, cache8 = relu_forward(out7)
        out9, cache9 = fc_forward(out8, W4, b4)
        scores = out9
    
        
        
        # raise NotImplementedError("TODO: Add your implementation here.")
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k].                                                      #
        # Hint: The backwards from W3 needs to be un-flattened before it can be    #
        # passed into the max pool backwards                                       #
        ############################################################################
        loss, dx = softmax_loss(scores, y)

        dx9, dw9, db9 = fc_backward(dx, cache9)
        dx8 = relu_backward(dx9, cache8)
        dx7_flat, dw7, db7 = fc_backward(dx8, cache7)

        dx7 = dx7_flat.reshape(N,F,H,W)

        dx6 = max_pool_backward(dx7, cache6)
        dx5 = relu_backward(dx6, cache5)
        dx4, dw4 = conv_backward(dx5, cache4)
        dx3 = max_pool_backward(dx4, cache3)
        dx2 = relu_backward(dx3, cache2)
        dx1, dw1 = conv_backward(dx2, cache1)

        grads = {'W1':dw1, 'W2':dw4, 'W3':dw7, 'W4':dw9, 'b3':db7, 'b4':db9}
        # raise NotImplementedError("TODO: Add your implementation here.")
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
