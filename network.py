import math
import numpy as np
#import cupy as np

activation_functiones = {
    'sigmoid': lambda x : math.exp(-np.logaddexp(0, -x)),
    'tanh' : lambda x : np.tanh(x)
}

class Network():
    def __init__(self,input_size,hidden_layers):
        self.input_size = input_size
        self.layers = hidden_layers

    def feed_forward(self,data):
        if self.input_size != len(data):
            raise ValueError("Wrong data size {} {}".format(self.input_size,len(data)))
        for layer in self.layers:
            data = layer.feed_forward(data)

        return data

    def train(self,input_data,input_labels):
        accuracy = 0
        average_cost = 0
        for image, label in zip(input_data,input_labels):
            output = self.feed_forward(image)
            cost = self._cost_function(output,self._generate_one_hot(label))
            average_cost += cost
            if np.argmax(output) == label:
                accuracy +=1

        return average_cost

    def _cost_function(self,input,desired_input):
        return np.sum(np.power(input-desired_input,2))

    def _generate_one_hot(self,input):
        one_hot = np.zeros(10)
        one_hot[input] = 1
        return one_hot



class Layer():
    def __init__(self,size,input_size,activation_function='sigmoid'):
        self.weights = np.random.rand(input_size,size)*0.01
        self.biases = np.zeros(shape=(size))
        self.activation_function = np.vectorize(activation_functiones[activation_function])

    def feed_forward(self,input):
        return self.activation_function(np.add(np.dot(input, self.weights),self.biases))
