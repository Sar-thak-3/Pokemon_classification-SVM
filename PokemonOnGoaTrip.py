# Image classification of pokemons

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.utils import load_img,img_to_array

p = Path("./Images")
train = pd.read_csv("train.csv")
train = train.values
# print(train[:5])

image_data = []
labels = []
pokemon_dict = {'Pikachu': 0,'Bulbasaur':1,"Charmander": 2}

for item in train:
    loadImg = load_img(Path(f"./Images/{item[0]}"),target_size=(32,32))
    image_data.append(img_to_array(loadImg))
    labels.append(pokemon_dict[item[1]])

# print(len(image_data))
# print(len(labels))

image_data = np.array(image_data,dtype='float32')/255.0
labels = np.array(labels)

# Randomly shuffling
import random
combined = list(zip(image_data,labels))
random.shuffle(combined)
image_data[:],labels[:] = zip(*combined)
# print(labels)
# print(type(image_data),labels.shape)

# DATA VISUALISATION
def visualizeImg(singleImg):
    plt.imshow(singleImg)
    plt.axis("off")
    plt.show()
# visualizeImg(image_data[0])

# SVM CLASSIFIER
class SVM:
    def __init__(self,c=1.0):
        self.c = c
        self.bias = 0
        self.weights = 0
    
    def hingeLoss(self,weights,x,y,bias=0):
        loss = 0.0
        loss += 0.5*np.dot(weights,weights.T)
        for i in range(x.shape[0]):
            ti = y[i]*(np.dot(weights,x[i].T)+bias)
            loss += self.c * max(0,(1-ti))
        return loss[0][0]
    
    def fit(self,x,y,batch_size=100,learning_rate=0.00001,maxItr=200):
        no_of_samples = x.shape[0]
        no_of_features = x.shape[1]

        w = np.zeros((1,no_of_features))
        losses = []
        bias = 0
        for i in range(maxItr):
            l = self.hingeLoss(w,x,y,bias)
            losses.append(l)
            ids = np.arange(no_of_samples)
            np.random.shuffle(ids)

            for batch_start in range(0,no_of_samples,batch_size):
                gradw = 0
                gradb = 0

                for j in range(batch_start,batch_start+batch_size):
                    if (j<no_of_samples):
                        id_no = ids[j]
                        ti = y[id_no]*(np.dot(w,x[id_no].T)+bias)
                        if(ti>=1):
                            pass
                        else:
                            gradb += (-1)*self.c*y[id_no]
                            gradw += (-1)*self.c*y[id_no]*x[id_no]
                    else:
                        break
                w = w - learning_rate*gradw
                bias = bias - learning_rate*gradb
        self.weights = w
        self.bias = bias
        return self.weights,self.bias,losses

# ONE vs ONE CLASSIFIER
image_data_shape = image_data.shape
temp_image_data = image_data.reshape((image_data_shape[0],-1))
# print(temp_image_data.shape)
# print(temp_image_data[0,:])
def getPairwiseData(x,y,class1,class2):
    data_pair = []
    data_labels = []
    for index,item in enumerate(y):
        if(item==class1):
            data_pair.append(x[index])
            data_labels.append(item)
        if(item==class2):
            data_pair.append(x[index])
            data_labels.append(item)
    return (np.array(data_pair),np.array(data_labels))

classes = np.unique(labels).shape[0]
# print(classes)

# Training NC2 SVM'S PART!
mysvm = SVM()
def trainSvm(x,y):
    svm_classifiers = {}
    for i in range(classes):
        svm_classifiers[i] = {}
        class1 = i
        for j in range(i+1,classes):
            class2 = j
            data_pair,data_labels = getPairwiseData(x,y,class1,class2)
            wts,bias,losses = mysvm.fit(data_pair,data_labels)
            svm_classifiers[i][j] = (wts,bias)
    return svm_classifiers

# print(temp_image_data[:2],labels[:2])
svm_classifiers = trainSvm(temp_image_data,labels)
print(svm_classifiers[0][1])

# PREDICTION
def binaryPredict(x,weights,bias):
    z = np.dot(x,weights.T) + bias
    if(z>=0):
        return 1
    else: 
        return -1

def predict(x):
    count = np.zeros((classes,))
    for i in range(classes):
        for j in range(i+1,classes):
            w,b = svm_classifiers[i][j]
            z = binaryPredict(x,w,b)
            print(z)
            if(z==1):
                count[j] += 1
            else:
                count[i] += 1
    # print(count)
    return max(count)

# Test data
test = pd.read_csv("test.csv")
test = test.values

# predict()  ->>>> 
def predict_test(test):
    for item in test:
        loadImg = load_img(Path(f"./Images/{item[0]}"),target_size=(32,32))
        x = img_to_array(loadImg)
        x = np.array(x,dtype='float32')/255.0
        x = x.reshape((1,-1))
        # print(x.shape)
        print(predict(x))
predict_test(test)
# predict_test(train)

# ACCURACY
def accuracy(x,y):
    count = 0
    for i in range(x.shape[0]):
        prediction = predict(x[i])
        if(prediction==y[i]):
            count+=1
    return count/x.shape[0]

# print(accuracy())