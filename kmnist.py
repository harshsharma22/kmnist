import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,Dense,MaxPool2D,Dropout,Activation,Flatten
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
train_data = "kuzushiji/kmnist-train-imgs.npz"
train_label = "kuzushiji/kmnist-train-labels.npz"

test_data = "kuzushiji/kmnist-test-imgs.npz"
test_label = "kuzushiji/kmnist-test-labels.npz"


train_data = np.load(train_data)['arr_0']
train_label = np.load(train_label)['arr_0']

test_data = np.load(test_data)['arr_0']
test_label = np.load(test_label)['arr_0']
#print(train_data.shape)

test_data,val_data,test_label,val_label = train_test_split(test_data,test_label,test_size=.10,random_state=42)

#train_label = to_categorical(train_label,10)
#test_label = to_categorical(test_label,10)
#val_label = to_categorical(val_label,10)
print(train_label.shape)
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
val_data = val_data.astype('float32')

train_data=train_data/255
test_data= test_data/255
val_data = val_data/255

shapes= (train_data.shape[1],train_data.shape[2],1)

train_data = train_data.reshape(train_data.shape[0],*shapes)
test_data = test_data.reshape(test_data.shape[0],*shapes)
val_data= val_data.reshape(val_data.shape[0],*shapes)




model = Sequential()
model.add(Conv2D(32,kernel_size=(5,5),input_shape=(28,28,1)))
model.add(Activation('relu'))


model.add(MaxPool2D(2,strides=(1,1)))
model.add(Dropout(.50))
model.add(Conv2D(64,kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(Conv2D(128,kernel_size=(2,2)))
model.add(Activation('relu'))
model.add(MaxPool2D(2,2))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(.50))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(.50))

model.add(Dense(10))
model.add(Activation('softmax'))

model.summary(120)

model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=['accuracy'])

history = model.fit(train_data,train_label,batch_size=32,epochs=2,validation_data=(val_data,val_label))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
label=model.predict_classes(test_data)
print(label)
print(test_label)
print(accuracy_score(test_label,label))

model.save_weights('mymodel3.h5')
json=model.to_json()

