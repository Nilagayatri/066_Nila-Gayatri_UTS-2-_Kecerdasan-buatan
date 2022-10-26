#Nama   : Nila Gayatri
#NIM    : 21091397066
#Kelas  : 2021 B
#Multi Neuron Batch Input 2 layer

#inisialisasi numpy
import numpy as np

#input layer feature 10
# per-batch ada 6 input
inputs = [
    [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.9, 4.5, 5.0, 5.5],
    [8.5, 1.4, 2.2, 2.4, 3.2, 4.4, 4.2, 3.4, 5.2, 5.4],
    [3.5, 18.5, 18.0, 20.5, 31.0, 30.5, 43.0, 40.5, 50.0, 50.5],
    [4.7, 5.8, 2.6, 9.8, 3.6, 5.8, 4.6, 4.8, 5.6, 5.8],
    [2.5, 6.4, 7.2, 7.4, 8.2, 8.4, 9.2, 9.4, 10.2, 10.4],
    [13.5, 17.4, 19.2, 17.4, 16.2, 15.4, 19.2, 10.4, 10.2, 11.4]]

# layer 1 = 5 neuron
weights = [
    [1.0, 1.5, 2.0, 2.5, 3.7, 4.5, 4.7, 4.5, 5.0, 5.5],
    [1.5, 1.4, 2.2, 3.4, 3.2, 3.4, 4.2, 4.4, 5.2, 5.4],
    [2.7, 2.8, 2.6, 2.8, 3.6, 3.8, 4.6, 4.8, 5.6, 5.8],
    [2.5, 6.4, 7.2, 7.4, 8.2, 8.4, 9.2, 9.4, 10.2, 10.4],
    [3.5, 18.5, 18.0, 20.5, 30.0, 30.5, 40.0, 40.5, 50.0, 50.5]]
biases1 = [1.5,3.0,3.8,4.5,9.2]

# layer 2 = 3 neuron
weights2 = [[1.5, 2.0, 3.0, 4.4, 9.2,],
            [2.5, 2.6, 7.4, 1.5, 2.0],
            [1.4, 2.2, 4.4, 4.8, 8.4]]
biases2 = [4.4, 2.5, 1.5] 

# Output layer 1
layer1_outputs = numpy.dot(inputs, numpy.array(weights1).T) + biases1

# Output layer 2
layer2_outputs = numpy.dot(layer1_outputs, numpy.array(weights2).T) + biases2

# Print Output layer 2
print(layer2_outputs) 