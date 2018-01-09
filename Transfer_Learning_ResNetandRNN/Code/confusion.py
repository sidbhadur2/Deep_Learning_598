from numpy import genfromtxt
import scipy.misc
import matplotlib.pyplot as plt

#data = genfromtxt('/Users/siddharthbhaduri/Desktop/Work/Fall-2017/IE-598/HW5/HW5_Action_Recognition/1.csv', delimiter=',')

#data = genfromtxt('/Users/siddharthbhaduri/Desktop/Work/Fall-2017/IE-598/HW5/HW5_Action_Recognition/2.csv', delimiter=',')

#data = genfromtxt('/Users/siddharthbhaduri/Desktop/Work/Fall-2017/IE-598/HW5/HW5_Action_Recognition/3.csv', delimiter=',')


print(data.shape)

data=data+128

plt.imshow(data,cmap='RdBu_r')
plt.show()