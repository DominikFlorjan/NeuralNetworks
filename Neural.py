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
        self.wih = np.random.normal(0.0, pow(self.hNodes, -0.5), (self.hNodes, self.inNodes)) 
        self.who = np.random.normal(0.0, pow(self.outNodes, -0.5), (self.outNodes, self.hNodes)) 

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
input_nodes = 784 #28 * 28 pixel size of an image
hidden_nodes = 200 # arbitrary number  
output_nodes = 10 # 10 possible outputs 0-9 
learning_rate = 0.1

# instance of neural network
neural = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Training
training_data_file = open("TrainingData/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for record in training_data_list:
    # data from csv to array
    all_values = record.split(',')
    # Scaling the input to be in range 0.1 - 1, we add 0.01 at the end because we if we didnt we would get a lot of zeroes
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = np.zeros(output_nodes) + 0.01 
    # all_values[0] is a label of this particular record so we set its target to max 
    targets[int(all_values[0])] = 0.99
    neural.train(inputs, targets)

# Testing
test_data_file = open("TrainingData/mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

all_values = test_data_list[0].split(',')
print("input: ", all_values[0])

# image_array = np.asfarray(all_values[1:]).reshape((28,28))
# ptl.imshow(image_array, cmap='Greys', interpolation='None')
# ptl.show()

input = neural.query((np.asfarray(all_values[1:])/255.0 * 0.99)+ 0.01)
print(input)