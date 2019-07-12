import network
import random
import math
import numpy as np
#import cupy as np
from mnist import MNIST

def main():
    mndata = MNIST('MNIST_data')

    mnist_images_train, mnist_labels_train = mndata.load_training()
    mnist_images_test, mnist_labels_test = mndata.load_testing()

    rested_size = 60000
    mnist_images_train = mnist_images_train[:rested_size]
    mnist_labels_train = mnist_labels_train[:rested_size]

    batch_size = 1000
    hm_epochs = 1

    input_layer_size = 784
    hidden_layer_size = 500
    output_layer_size = 10

    nn = network.Network(
            input_layer_size,
            [
                network.Layer(hidden_layer_size,input_layer_size,'tanh'),
                network.Layer(hidden_layer_size,hidden_layer_size,'tanh'),
                network.Layer(output_layer_size,hidden_layer_size)
            ]
        )

    train(nn,mnist_images_train,mnist_labels_train,batch_size,hm_epochs)




def train(network,train_images,train_labels,batch_size,hm_epochs):
    for epoch in range(hm_epochs):
        batch_count = round(len(train_images)/batch_size)
        c_average = 0
        for batch in range(batch_count):
            c = network.train(
                train_images[
                    batch*batch_size:batch*batch_size+batch_size
                    ],
                train_labels[
                    batch*batch_size:batch*batch_size+batch_size
                    ])

            c_average += c*len(train_labels[
                                batch*batch_size:batch*batch_size+batch_size
                                ])
            print('Batch {} out of {} finished'.format(batch+1,batch_count))

        print('Epoch {} completet out of {} cost:\t{c}'.format(
            epoch+1,
            hm_epochs,
            c_average/len(train_images)
            ))



main()
