import numpy as np
arr1 = []

with open('predictions_1.txt') as file:
    for line in file.readlines():
        arr1.append(line.strip().split(','))
        
dic ={}
for i in range(len(arr1)):
    dic[arr1[i][0][2:-1]] = float(arr1[i][1].strip())
    
arr2 = []
with open('combined_predictions.txt') as file:
    for line in file.readlines():
        arr2.append(line.strip().split(','))
        
dic2={}

for i in range(len(arr2)):
    dic2[arr2[i][0][2:-1]] = float(arr2[i][1].strip())
    
new_dic ={}


for i in dic2.keys():
    new_dic[i] = dic[i]-dic2[i]
    
import pandas as pd
df = pd.DataFrame.from_dict(new_dic, orient = 'index')

df.to_csv('Difference_in_accuracy.csv')
