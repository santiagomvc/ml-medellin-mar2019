#%%
#------------importing packages-------------------#
import tensorflow as tf
import scipy
from PIL import Image
from scipy import ndimage
import numpy as np
import os
import math
import h5py
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import pandas as pd

#%%
# defining folders path for training
train = 'C:\\ml-medellin-mar2019\\train'

#%%
#read info from training
X_train = []
Y_train = []
for i in os.listdir(train):
    label = i
    for j in  os.listdir(train + '\\' + i):
        fname = train + '\\' + i + '\\' + j
        image = np.array(ndimage.imread(fname, flatten=False))
        reshaped_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
        X_train.append(reshaped_image)
        Y_train.append(float(label))

#%%
# Preprocessing training data

X_train = np.array(X_train)
X_train = X_train.squeeze()
X_train = X_train.T

Y_train = np.array(Y_train)
Y_train = Y_train.reshape(1, -1)

# Normalizing the inputs
X_train = X_train/255.

#%%
# Convert training and test labels to one hot matrices
temp_Y_train = Y_train.flatten().astype(int)
Y_train = np.zeros((temp_Y_train.size, np.max(temp_Y_train)+1))
Y_train[np.arange(temp_Y_train.size), temp_Y_train] = 1
Y_train = Y_train.T

#%%
# separando dev and test
dev_size = math.ceil(0.15 * X_train.shape[0])
dev_index = np.random.randint(0, X_train.shape[0], dev_size)
X_test = X_train[:, dev_index]
Y_test = Y_train[:, dev_index] 
X_train = np.delete(X_train, dev_index, axis = 1)
Y_train = np.delete(Y_train, dev_index, axis = 1)

#%%
#             Creating the functions for the NN

#FUNCTION: create_placeholders
def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    ### START CODE HERE ### (approx. 2 lines)
    X = tf.placeholder(tf.float64, shape=(n_x, None))
    Y = tf.placeholder(tf.float64, shape=(n_y, None))
    ### END CODE HERE ###
    
    return X, Y


# X, Y = create_placeholders(X_train.shape[0], Y_train.shape[0])
# print ("X = " + str(X))
# print ("Y = " + str(Y))


# initialize_parameters
def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. 
    
    Returns:
    parameters -- a dictionary of tensors containing 
    """
    
    tf.set_random_seed(1)                   # so that your "random" numbers match ours
        
    ### START CODE HERE ### (approx. 6 lines of code)
    W1 = tf.get_variable('W1', [1000, 12288], initializer= tf.contrib.layers.xavier_initializer(seed = 1), dtype = tf.float64)
    b1 = tf.get_variable('b1', [1000, 1], initializer= tf.zeros_initializer(), dtype = tf.float64)
    W2 = tf.get_variable('W2', [750, 1000], initializer= tf.contrib.layers.xavier_initializer(seed = 1), dtype = tf.float64)
    b2 = tf.get_variable('b2', [750, 1], initializer= tf.zeros_initializer(), dtype = tf.float64)
    W3 = tf.get_variable('W3', [500, 750], initializer= tf.contrib.layers.xavier_initializer(seed = 1), dtype = tf.float64)
    b3 = tf.get_variable('b3', [500, 1], initializer= tf.zeros_initializer(), dtype = tf.float64)
    W4 = tf.get_variable('W4', [250, 500], initializer= tf.contrib.layers.xavier_initializer(seed = 1), dtype = tf.float64)
    b4 = tf.get_variable('b4', [250, 1], initializer= tf.zeros_initializer(), dtype = tf.float64)
    W5 = tf.get_variable('W5', [125, 250], initializer= tf.contrib.layers.xavier_initializer(seed = 1), dtype = tf.float64)
    b5 = tf.get_variable('b5', [125, 1], initializer= tf.zeros_initializer(), dtype = tf.float64)
    W6 = tf.get_variable('W6', [80, 125], initializer= tf.contrib.layers.xavier_initializer(seed = 1), dtype = tf.float64)
    b6 = tf.get_variable('b6', [80, 1], initializer= tf.zeros_initializer(), dtype = tf.float64)
    W7 = tf.get_variable('W7', [43, 80], initializer= tf.contrib.layers.xavier_initializer(seed = 1), dtype = tf.float64)
    b7 = tf.get_variable('b7', [43, 1], initializer= tf.zeros_initializer(), dtype = tf.float64)
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5,
                  "W6": W6,
                  "b6": b6,
                  "W7": W7,
                  "b7": b7}
    
    return parameters


# tf.reset_default_graph()
# with tf.Session() as sess:
#     parameters = initialize_parameters()
#     print("W1 = " + str(parameters["W1"]))
#     print("b1 = " + str(parameters["b1"]))
#     print("W2 = " + str(parameters["W2"]))
#     print("b2 = " + str(parameters["b2"]))

# forward_propagation

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']
    W6 = parameters['W6']
    b6 = parameters['b6']
    W7 = parameters['W7']
    b7 = parameters['b7']
    

    
    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1,X),b1)                                              # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)                                              # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)                                              # Z3 = np.dot(W3,Z2) + b3
    A3 = tf.nn.relu(Z3)                                              # A2 = relu(Z2)
    Z4 = tf.add(tf.matmul(W4,A3),b4)                                              # Z3 = np.dot(W3,Z2) + b3
    A4 = tf.nn.relu(Z4)                                              # A2 = relu(Z2)
    Z5 = tf.add(tf.matmul(W5,A4),b5)                                              # Z3 = np.dot(W3,Z2) + b3
    A5 = tf.nn.relu(Z5)                                              # A2 = relu(Z2)
    Z6 = tf.add(tf.matmul(W6,A5),b6)                                              # Z3 = np.dot(W3,Z2) + b3
    A6 = tf.nn.relu(Z6)                                              # A2 = relu(Z2)
    Z7 = tf.add(tf.matmul(W7,A6),b7)                                              # Z3 = np.dot(W3,Z2) + b3
    ### END CODE HERE ###
    
    return Z7

# tf.reset_default_graph()
# with tf.Session() as sess:
#     X, Y = create_placeholders(X_train.shape[0], Y_train.shape[0])
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     print("Z3 = " + str(Z3))


# Computing Cost
def compute_cost(Z7, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z7)
    labels = tf.transpose(Y)
    
    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    ### END CODE HERE ###
    
    return cost

# tf.reset_default_graph()
# with tf.Session() as sess:
#     X, Y = create_placeholders(X_train.shape[0], Y_train.shape[0])
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     cost = compute_cost(Z3, Y)
#     print("cost = " + str(cost))


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
        
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    ### START CODE HERE ### (1 line)
    X, Y = create_placeholders(n_x, n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters()
    ### END CODE HERE ###

    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z7 = forward_propagation(X, parameters)
    ### END CODE HERE ###

    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z7, Y)
    ### END CODE HERE ###

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    ### END CODE HERE ###

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        num_epoc_actual = 0

        # Do the training loop
        for epoch in range(num_epochs):
            num_epoc_actual = num_epoc_actual + 1
            print(num_epoc_actual)

            epoch_cost = 0.                       # Defines a cost related to an epoch
            
        
            
            # IMPORTANT: The line that runs the graph on a minibatch.
            # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
            ### START CODE HERE ### (1 line)
            _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
            ### END CODE HERE ###
            
            epoch_cost += minibatch_cost 

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z7), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters

#%%
# Running the model
parameters = model(X_train, Y_train, X_test, Y_test)

#%%
# Predicting on the test set
test = 'C:\\ml-medellin-mar2019\\test_files\\test_files'

#%%
#read info from training
X_prove = []
X_prove_name = []
for i in os.listdir(test):
    fname = test + '\\' + i
    image = np.array(ndimage.imread(fname, flatten=False))
    reshaped_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
    X_prove.append(reshaped_image)
    X_prove_name.append(i)
 #   Y_train.append(float(label))

X_prove = np.array(X_prove)
X_prove = X_prove.squeeze()
X_prove = X_prove.T

# Normalizing the inputs
X_prove = X_prove/255.

#%%
# with tf.Session() as sess:
#     #X, Y = create_placeholders(X_prove.shape[0], 0)
#     response = sess.run(forward_propagation(X_prove, parameters))
#     response = sess.run(tf.nn.softmax(response))

tf.reset_default_graph()
with tf.Session() as sess:
    X, Y = create_placeholders(X_prove.shape[0], Y_train.shape[0])
#     parameters = initialize_parameters()
    response = forward_propagation(X_prove, parameters)
    response = sess.run(tf.nn.softmax(response))

#%%
real_response = np.argmax(response, axis = 0)


#%%
df_results = pd.DataFrame({'file_id': X_prove_name, 'label': real_response})
test_csv = pd.read_csv('C:\\ml-medellin-mar2019\\test.csv')
test_csv = test_csv.drop('label', axis = 1)
definitive_results = pd.merge(test_csv, df_results, on = 'file_id', how = 'outer')
definitive_results.to_csv('C:\\ml-medellin-mar2019\\test_results.csv', index=False)
#%%
