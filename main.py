# Import necessary packages

import allel
import pandas as pd
import numpy as np

# Convert VCF to CSV file - DONE
#allel.vcf_to_csv('chromosom22.vcf', 'chromosom22.csv')

# Load our data as .csv file

data = pd.read_csv('chromosom22.csv')

######### Data transformation #########

# DataFrame transponse

data = data.T

# print length of DataFrame (SNP = 1103548)

total_rows = data.count()
print(total_rows + 1)

# drop unused columns: QUAL, FILTER_PASS, ID

#columns = ['QUAL', 'FILTER_PASS', 'ID']

#data.drop(columns, axis=1, inplace=True)

# drop unused rows: ALT_2, ALT_3

#df1 = data[data['ALT_2'].str.len() > 0]
#df2 = data[data['ALT_3'].str.len() > 0]

print(data)



# PCA (ang. Principal Component Analysis) - Analiza głównych składowych

#from sklearn.decomposition import PCA
