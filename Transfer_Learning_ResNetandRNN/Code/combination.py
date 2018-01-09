import numpy as np
import os
import time

from multiprocessing import Pool

from helperFunctions import getUCF101
import h5py


acc_top1 = 0.0 
acc_top5 = 0.0
acc_top10 = 0.0

num_classes = 101

confusion_matrix = np.zeros((101,101),dtype=np.float32)

data_directory = '/u/training/tra044/scratch/HW5/HW5_Action_Recognition/'
class_list, train, test = getUCF101(base_directory = data_directory)

for i in range(len(test[0])):
	label = test[1][i]
	filename = test[0][i]
	filename = filename.replace('.avi','.hdf5')
	filename = filename.replace('UCF-101','UCF-101-predictions')

	if(not os.path.isfile(filename)):
		continue
	with h5py.File(filename,'r') as h:
		pred = h['predictions'][:]
	pred = np.mean(pred,axis=0)

	filename2 = filename.replace('UCF-101-predictions','UCF-101-predictions-2')
	if(not os.path.isfile(filename2)):
		continue
	
	with h5py.File(filename2,'r') as h:
		pred2 = h['predictions'][:]
	pred2 = np.mean(pred2,axis=0)

	pred_combined = (pred + pred2)/2

	if (not pred_combined.shape[0] == 101):
   		pred_combined = pred_combined.transpose()

	argsort_pred = np.argsort(-pred_combined)[0:10]

	confusion_matrix[label,argsort_pred[0]] += 1
	if(label==argsort_pred[0]):
		acc_top1 += 1.0
	if(np.any(argsort_pred[0:5]==label)):
		acc_top5 += 1.0
	if(np.any(argsort_pred[:]==label)):
		acc_top10 += 1.0
	print('i:%d (%f,%f,%f)' 
		% (i,acc_top1/(i+1),acc_top5/(i+1), acc_top10/(i+1)))

number_of_examples = np.sum(confusion_matrix,axis=1)

fig_dir = '/u/training/tra044/scratch/HW5/HW5_Action_Recognition/'

file_name = str(3) + ".csv"
file_loc = fig_dir + file_name

for i in range(num_classes):
	confusion_matrix[i,:] = confusion_matrix[i,:]/np.sum(confusion_matrix[i,:])

np.savetxt(file_loc, confusion_matrix, delimiter=",")

results = np.diag(confusion_matrix)
indices = np.argsort(results)

sorted_list = np.asarray(class_list)
sorted_list = sorted_list[indices]
sorted_results = results[indices]

for i in range(num_classes):
	print(sorted_list[i],sorted_results[i],number_of_examples[indices[i]])






