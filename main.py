# Import necessary packages

import allel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nn import neural_network

# Load our data as .csv file

dataset = pd.read_csv('chr22qc_example.csv') # A" (m x n)


######### Data transformation #########

"""
# Drop unused rows

with open("chromosom22.csv") as input_file:
    with open("output.csv", "w") as output_file:
        for i, line in enumerate(input_file.readlines()):
            if i == 0:
                output_file.write(line)
                continue
            CHROM, POS, ID, REF, ALT_1, \
            ALT_2, ALT_3, QUAL, FILTER_PASS = line.split(",")
            if ALT_2 or ALT_2:
            output_file.write(line)

outData = pd.read_csv('output.csv')

# Drop unused columns : QUAL, FILTER_PASS, ID, ALT_2, ALT_3

columns = ['QUAL', 'FILTER_PASS', 'ID', 'ALT_2', 'ALT_3']

outData.drop(columns, axis=1, inplace=True) # now A'(m x k)


# print length of DataFrame (SNP = 1097200) = ALT1 - ALT_2 - ALT_3

print(outData)

# Save Data into DataFrame .csv

outData = outData.to_csv('afterDrop.csv', index=0)

"""

# print length of Data

print(dataset) # 2504 rows x 11159 columns

# PCA


#neural_network()