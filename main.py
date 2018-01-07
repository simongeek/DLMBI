# Import necessary packages

import allel
import pandas as pd
import numpy as np

# Convert VCF to CSV file - DONE
#allel.vcf_to_csv('chromosom22.vcf', 'chromosom22.csv')

# Load our data as .csv file

data = pd.read_csv('chromosom22.csv')

######### Data transformation #########

# Drop unused columns : QUAL, FILTER_PASS, ID

columns = ['QUAL', 'FILTER_PASS', 'ID']

data.drop(columns, axis=1, inplace=True)

# DataFrame transponse A( m x n)

#data = data.T

#out = data.to_csv('poTransponowaniu.csv', index=False)

# print length of DataFrame (SNP = 1103548)

total_rows = data.count()
print(total_rows + 1)

# drop unused rows: ALT_2, ALT_3


# PCA (ang. Principal Component Analysis) - Analiza głównych składowych

#from sklearn.decomposition import PCA
