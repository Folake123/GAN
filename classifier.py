# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 19:17:08 2018

@author: flakk
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import random 
import dask.dataframe as dd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import json
from keras.models import model_from_json, load_model
seed=7
np.random.seed(seed)

# 1.Reading the CSV file
#mydt = pd.read_csv("./output_20181102012430.csv")
'''df1 = dd.read_csv('/depot/datalab/sakhala/data/clean_data/output1.csv',dtype=np.float64)
print("file1 read",len(df1),df1.shape)
df=df1
df2 = dd.read_csv('/depot/datalab/sakhala/data/clean_data/output2.csv',dtype=np.float64)
print("file2 read",len(df2),df2.shape)
df.append(df2)
df3 = dd.read_csv('/depot/datalab/sakhala/data/clean_data/output3.csv',dtype=np.float64)
print("file3 read",len(df3),df3.shape)
df.append(df3)
df4 = dd.read_csv('/depot/datalab/sakhala/data/clean_data/output4.csv',dtype=np.float64)
print("file4 read",len(df4),df4.shape)
df.append(df4)
df5 = dd.read_csv('/depot/datalab/sakhala/data/clean_data/output5.csv',dtype=np.float64)
print("file5 read",len(df5),df5.shape)
df.append(df5)
df6 = dd.read_csv('/depot/datalab/sakhala/data/clean_data/output6.csv',dtype=np.float64)
print("file6 read",len(df6),df6.shape)
df.append(df6)
#frames = [df1,df2,df3,df4,df5,df6]
#df = dd.concat(frames)
print("concat - total size: ",df.shape,len(df))'''
df = pd.read_csv("/scratch/rice/n/nsakhala/sampled.csv")
#df.columns=['l1','l2','l3','l4','l5','l6','l7','l8','l9','l10','l11','l12','label']
df.columns=['year','month','day','hour','minute','second','duration','source_ip','dest_ip','source_port','dest_port','protocol','flag1','flag2','flag3','flag4','flag5','flag6','fwd','stos','pkt','byt','label']
print(df.info())
'''
criteria0 =  df['label'] == 0
criteria1 = df['label'] == 1
df = df.dropna()
print("after nulls: ",len(df))
data0 = df[criteria0]
data1 = df[criteria1]
print("shape of labels 0 is: ",len(data0))
print("shape of label 1 is: ",len(data1))

frac0 = 0.9
data0 = data0.sample(frac=frac0)
frac1 = (6*len(data0))/len(data1)
data1 = data1.sample(frac=frac1)
print("sampled shape of labels 0 is: ",len(data0))
print("sampled shape of label 1 is: ",len(data1))

frames = [data0, data1]
final = dd.concat(frames)
print("shape of merged data: ",len(final))
'''
final = df.sample(frac=1).reset_index(drop=True)
print("size of dataset - ",len(final))


# 2.convert the columns value of the dataset as floats
float_array = final.values.astype(float)
 
# 3. create a min max processing object
min_max_scaler = MinMaxScaler()
scaled_array = min_max_scaler.fit_transform(float_array)

# 4. convert the scaled array to dataframe
mydt_normalized = pd.DataFrame(scaled_array)
mydt_normalized
#print(mydt_normalized)

#  Create matrix of features and matrix of target variable
X = mydt_normalized.iloc[:, 0:22].values
y = mydt_normalized.iloc[:, 22].values

print("X is: ",X.shape,X.size)
print(y[0:3])

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for index, (train_indices, val_indices) in enumerate(kfold.split(X, y)):
    xtrain, xval = X[train_indices], X[val_indices]
    ytrain, yval = y[train_indices], y[val_indices]
# Splitting the dataset into the Training set and Test set i used 80:20
    print("train is: ",xtrain.shape,ytrain.shape)
    print("for test:",xval.shape,yval.shape)
# Importing the Keras libraries and packages
#Initializing Neural Network
    classifier=None
    classifier = Sequential()
# Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim = 22, init = 'normal', activation = 'relu', input_dim = 22))
# Adding the second hidden layer
    #classifier.add(Dense(output_dim = 22, init = 'normal', activation = 'relu'))
# Adding the output layer
#classifier.add(Dense(output_dim = 500, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'normal', activation = 'sigmoid'))
# Compiling Neural Network
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fit the model
#classifier.fit(X,y, epochs = 20, batch_size =10)
    history = classifier.fit(xtrain, ytrain,shuffle=True,  epochs=20, batch_size=10000,validation_data=(xval,yval))
    #print("%s: %.2f%%" % (classifier.metrics_names[1], history.history['val_acc']*100))
    #cvscores.append(history.history['val_acc'] * 100)  

#print("total - %.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
#Evaluate the model
#scores = classifier.evaluate(X,y)
#print("\n%s: %.2f%%" %(classifier.metrics_names[1], scores[1]*100))

# Predicting the Test set results
#y_pred = classifier.predict(X_test)
#y_pred = (y_pred > 0.5)
# Creating the Confusion Matrix

#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred)
print ("confusion_matrix\n",cm)


#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
import matplotlib.pyplot as plt  
 
# code for building your model  
 
# train your model  
  
print(history.history.keys())  
   
plt.figure(1)  
   
# summarize history for accuracy  
   
plt.subplot(211)  
plt.plot(history.history['acc'])  
plt.plot(history.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
   
# summarize history for loss  
   
plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
plt.savefig('plots.png')
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


classifier.save_weights('model_weights.h5')
with open('model_architecture.json', 'w') as f:
    f.write(classifier.to_json())
