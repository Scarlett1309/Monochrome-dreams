#Importing libraries
import numpy as np #for linear algebra
import pandas as pd #for data processing
import keras
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D, BatchNormalization
from keras.preprocessing import image
from tqdm import tqdm
from tensorflow.keras import layers
from sklearn.metrics import hamming_loss
from sklearn.metrics import f1_score, accuracy_score,confusion_matrix, classification_report
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

#Reading the data

train_samples=pd.read_csv("train.txt", sep=',', header=None, names=["Data", "Label"])
validation_samples=pd.read_csv("validation.txt", header=None, sep=',', names=["Data", "Label"])
test_samples=pd.read_csv("test.txt", header=None)
sub_sample=pd.read_csv("sample_submission.txt")
test_data = np.array(test_samples.iloc[:, 0])
train_data=np.array(train_samples.iloc[:, 0])
train_label=np.array(train_samples.iloc[:, 1])
validation_datas=np.array(validation_samples.iloc[:, 0])
validation_label=np.array(validation_samples.iloc[:, 1])

#Exploring and preparing the data
test_image = []
for i in tqdm(range(test_data.shape[0])):
    img = image.load_img('Images/test/' + test_data[i], target_size=(28, 28, 3))
    img = image.img_to_array(img)
    img = img.astype('float32')
    img = img/255.0
    test_image.append(img)
X_test = np.array(test_image)

train_image = []
for i in tqdm(range(train_data.shape[0])):
    img = image.load_img('Images/train/' + train_data[i], target_size=(28, 28, 3))
    img = image.img_to_array(img)
    img=img/255.0
    img = img.astype('float32')
    train_image.append(img)
X_train=np.array(train_image)

validation_image = []
for i in tqdm(range(validation_datas.shape[0])):
    img = image.load_img('Images/validation/' + validation_datas[i], target_size=(28, 28, 3))
    img = image.img_to_array(img)
    img = img.astype('float32')
    img = img/255.0
    validation_image.append(img)
X_validation=np.array(validation_image)

train_label1 = pd.get_dummies(train_label).values
validation_label1 = pd.get_dummies(validation_label).values

#Creating the model
model= Sequential()
model.add(Conv2D(kernel_size=(3,3), padding ='same', filters=32, activation='tanh', input_shape=(28,28,3)))
model.add(Conv2D(filters=30,kernel_size = (3,3),activation='tanh'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(filters=30,kernel_size = (3,3),activation='tanh'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(filters=30,kernel_size = (3,3),activation='tanh'))

model.add(Flatten())

model.add(Dense(20,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(9,activation = 'softmax'))
    
model.compile(
              loss='categorical_crossentropy', 
              metrics=['accuracy'],
              optimizer='adam'
             )
#Training the data
model.fit(X_train,train_label1, batch_size=128, epochs=20)

#Evaluating our model
score = model.evaluate(X_validation, validation_label1, verbose=0)
print('Test loss : ', score[0])
print('Test accuracy: ', score[1])

validation_predict = model.predict(X_validation)
validation_predict = np.argmax(validation_predict, axis=1)


print(f"Accuracy score : {accuracy_score(validation_label, validation_predict)}")


#Confusion matrix

result = confusion_matrix(validation_label, validation_predict)
sn.heatmap(result, annot=True)
plt.show()

#Classification report
print("Classification report : \n", classification_report(validation_label, validation_predict))

#Making the preduction on testing data
prediction = model.predict(X_test)
prediction = np.argmax(prediction, axis=1)

#Creating the submission file
with open('submission.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(["id", "label"])
    for i in range(0, len(test_data)):
        wr.writerow([test_data[i], prediction[i]])
        
#------------------------------------
#Testing other models

#reshaping our data from 4D to 2D
nsamples, nx, ny, nz = X_train.shape
x = X_train.reshape((nsamples,nx*ny*nz))

nsamples, nx, ny, nz = X_validation.shape
xv = X_validation.reshape((nsamples,nx*ny*nz))


#Decision tree

dtree_model = DecisionTreeClassifier().fit(x, train_label)
val_pred = dtree_model.predict(xv)
print(f"Accuracy score : {accuracy_score(validation_label, val_pred)}")

print("Classification report : \n", classification_report(validation_label, val_pred))

#SGD 

sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(x, train_label)

v_pred = sgd_clf.predict(xv)

print(f"Accuracy score : {accuracy_score(validation_label, v_pred)}")
print("Classification report : \n", classification_report(validation_label, v_pred))

#Random Forest

rf_model = RandomForestClassifier(n_estimators = 100)
rf_model.fit(x, train_label)
vp = rf_model.predict(xv)

print(f"Accuracy score : {accuracy_score(validation_label, vp)}")

print("Classification report : \n", classification_report(validation_label, vp))

# --------------------------
#Final results 
'''
|     | Precision | Recall | f1 | Accuracy | 
| --- | --- | --- | --- | --- |
| CNN | 0.81 | 0.81  | 0.81  | 0.8096
| --- | --- | --- | --- | --- |
| Decision Tree  | 0.36 | 0.36 | 0.36 | 0.36 
| --- | --- | --- | --- | --- |
|Random Forest|0.62|0.62|0.62|0.6208
| --- | --- | --- | --- | --- |
|SGD| 0.58 | 0.55 | 0.55 | 0.55
'''