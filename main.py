# Import necessary packages

import allel
import pandas as pd
import numpy as np

# Convert VCF to CSV file - DONE
#allel.vcf_to_csv('chromosom22.vcf', 'chromosom22.csv')

# Load our data as .csv file

data = pd.read_csv('chromosom22.csv')


######### Data transformation #########


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
                continue
            output_file.write(line)

outData = pd.read_csv('output.csv')

# Drop unused columns : QUAL, FILTER_PASS, ID, ALT_2, ALT_3

columns = ['QUAL', 'FILTER_PASS', 'ID', 'ALT_2', 'ALT_3']

outData.drop(columns, axis=1, inplace=True)

# DataFrame transponse A( m x n)

outData = outData.T # we get A' (m x k)

out = outData.to_csv('poTransponowaniu.csv', index=False)

# print length of DataFrame (SNP = 1097200) = ALT1 - ALT_2 - ALT_3

print(outData)

# PCA - reduce matrix dimension

#from sklearn.decomposition import PCA
