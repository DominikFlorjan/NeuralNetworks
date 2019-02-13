import matplotlib.pyplot as ptl
import numpy as np
import scipy.special

#Main class
class NeuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        #Set numbers of nodes in each layer
        self.inNodes = inputNodes
        self.hNodes = hiddenNodes
        self.outNodes = outputNodes

        #Set learing rate
        self.lr = learningRate

        # Weights wih -- Weight_input_hidden
        # who -- weight hidden output
        # Using numpy gaussian normal distribution in point 0.0 with standard dev in second argument and size of matrix in third 
        # Rozbieżnośc w książce i na githubie !!!!
        self.wih = np.random.normal(0.0, pow(self.inNodes, -0.5), (self.hNodes, self.inNodes)) 
        self.who = np.random.normal(0.0, pow(self.hNodes, -0.5), (self.outNodes, self.hNodes)) 

        # Activation function (sigmoid/expit in scipy package)
        self.activation_function = lambda x : scipy.special.expit(x)

    #Two parts: Working out the output (basicly what is done in query) and comparing output to desired output
    def train(self, inputs_list, targets_list):
        # Inputs to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

       # Look up comments in query function 
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # Using the formula errors_hidden = weights.T_h-o times errors_outputs
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        # Updating the weight between nodes
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))


    def query(self, input_list):
        # Inputs to 2d array
        inputs = np.array(input_list, ndmin=2).T

        # Calc signals going into hidden layer 
        hidden_inputs = np.dot(self.wih, inputs)
        # Calc signals emerging from hidden layer 
        hidden_outputs = self.activation_function(hidden_inputs)
        # Calc signals going into final output layer  
        final_inputs = np.dot(self.who, hidden_outputs)
        # Calc signals emerging from final output layer 
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

#Parameters 
input_nodes = 3
hidden_nodes = 3 
output_nodes = 3
learning_rate = 0.3

#instance of neural network
neural = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Test
print(neural.query([1.0, 0.5, -1.5]))