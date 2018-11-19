# Library inclusions
import numpy as np
import math
from sklearn.model_selection import train_test_split # to be used to split the data in two parts


# Load the file and check the diemntions of the matrix
ionosphere = np.genfromtxt("ionosphere.txt",delimiter=',')
print("The matrix dimention of the data is: ", ionosphere.shape, ".\n")

# Split the data into four parts. X(Parameters) and Y(Labels) for both test and train.

#Note: 34th column of the data is the Y label. 
ionosphere_X_train, ionosphere_X_test, ionosphere_y_train, ionosphere_y_test = train_test_split(ionosphere[:,0:34], ionosphere[:,34], random_state=0)

# Check the size of the matrices hence created.
print("The size of the train and test data params are: ",ionosphere_X_train.shape,ionosphere_X_test.shape)
print("The size of the train and test data labels are:",ionosphere_y_train.shape,ionosphere_y_test.shape)


# Custom function for Euclidian Distance
def euc_dist(a,b):
    if (a.shape[0] != b.shape[0]):
        print("Bad Value")
    else: 
        dim = a.shape[0]
        s=0
    for i in range(0,dim):
        s+= (a[i] - b[i])**2
        #print(s)
    return np.sqrt(s)

#bubble sort
def bubbleSort(alist):
    for passnum in range(len(alist)-1,0,-1):
        for i in range(passnum):
            if alist[i]>alist[i+1]:
                temp = alist[i]
                alist[i] = alist[i+1]
                alist[i+1] = temp

from collections import Counter    
# K-Nearest Neighbour
def knn(k,train_x,test_x,train_y,test_y):
    n_test = test_x.shape[0]
    n_train = train_x.shape[0]
    # an array to store the distances from the test set to train set
    distances = np.zeros([n_train,n_test])
    sorted_distances = np.zeros([n_train,n_test])
    
    # an array to store the position of k nearest train neighbours
    pos=np.zeros([k,n_test])
    # an array to store predicted labels for test data
    nn_labels = np.zeros([k,n_test])
    pred = np.zeros([n_test])
    # looping over dataset to get all the distances
    for i in range(0,n_test):
        for j in range(0,n_train):
            distances[j,i] = euc_dist(train_x[j,],test_x[i,])
            sorted_distances[j,i] = euc_dist(train_x[j,],test_x[i,])
    
    #find the shortest distances and thier positions in array
    # sort the array
    a=0
    for i in range(0,n_test):
        bubbleSort(sorted_distances[:,i])
        for kk in range(0,k):
            pos[kk,i] = np.where(distances[:,i] == sorted_distances[kk,i])[0][0]
            nn_labels[kk,i] = train_y[int(pos[kk,i])] 
    #print(pos)
    
    # find apt labels
    acc_count = 0
    for t in range(0,n_test):
        cnt=Counter()
        for l in nn_labels[:,t]:
            cnt[l] += 1
        pred[t] = max(cnt, key=cnt.get)
    acc_count=np.sum(pred==test_y);
    return acc_count/n_test         


# Run the function for K=1 and K=3. 
ionosphere_k_1 = knn(1,ionosphere_X_train, ionosphere_X_test, ionosphere_y_train, ionosphere_y_test)
ionosphere_k_3 = knn(3,ionosphere_X_train, ionosphere_X_test, ionosphere_y_train, ionosphere_y_test)


# Print the results
print("Nearest Neighbour! \n")
print("IONOSPHERE DATASET \n")
print("K = 1. Accuracy ratio is: %2.5f \n"%ionosphere_k_1)
print("K = 3. Accuracy ratio is: %2.5f \n"%ionosphere_k_3)




