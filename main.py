# Import necessary packages

import pandas as pd
import sys
from nn import neural_network

# Load our data as .csv file
dataset = pd.read_csv('chr22qc_example.csv') # A" (m x n)



if (len(sys.argv) < 1):
    print("Please provide at least one hidden layer neurons number")
    exit(1)

hidden_neurons = []
for i in range(1, len(sys.argv)):
    hidden_neurons.append(int (sys.argv[i]))
print("Run neural network")
neural_network(dataset, hidden_neurons)

#print(dataset) # 2504 rows x 11159 columns
