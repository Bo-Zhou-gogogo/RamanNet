#强制使用CPU训练
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
time_start=time.time()

import numpy as np
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.models import load_model
#GPU
import tensorflow as tf
import keras
#GPU training
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


# 3000 finetune data
X = np.load('E:\\zhoubo\\data\\nc30\\X_finetune.npy')
y = np.load('E:\\zhoubo\\data\\nc30\\y_finetune.npy')
print(X.shape, y.shape)

i = 0
ave_accuracy=0
simple_pre=[]
simple_true=[]
#设置跑的epoch
nb_epoch =30
nb_features = 1000
nb_class = 30
ten_accuracy = []
random_state=0
nb_fold=5

for i in range (5):
    p_val = 0.1
    n_val = int(len(y) * p_val)
    idx_tr = list(range(len(y)))
    np.random.shuffle(idx_tr)
    idx_val = idx_tr[:n_val]
    idx_tr = idx_tr[n_val:]
    X_train, X_val, y_train, y_val = X[idx_tr], X[idx_val], y[idx_tr],y[idx_val]

    y_train = np_utils.to_categorical(y_train, nb_class)  #train set。size of (nb_samples, nb_classes)
    y_val = np_utils.to_categorical(y_val, nb_class)      #val set hot matrix

    #reshape train data, (sample numbers, channel size, channel numbers)
    X_train_r = np.zeros((len(X_train), nb_features, 1))
    X_train_r[:, :, 0] = X_train[:, :nb_features]

    #reshape val data
    X_val_r = np.zeros((len(X_val), nb_features, 1))
    X_val_r[:, :, 0] = X_val[:, :nb_features]

    #load the pre-trained model and set up sgd optimizer
    # model = load_model('model_pre_trained.hdf5')
    model = load_model('./model_pre_trained.hdf5')
    # sgd = keras.optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
 
    model.summary()

    # model save path
    filepath_log=os.path.join(os.path.join('./model'),'finetuned_model_test'+str(i)+'.hdf5')
    # monitor val_acc and save best model
    checkpoint=ModelCheckpoint(filepath_log, monitor='val_acc',verbose=1,save_best_only=True,mode='auto',period=1)
    # monitor val_acc and early stopping
    early_stop= EarlyStopping(monitor='val_acc', verbose = 1, min_delta=0.00001, patience=200, mode ='auto')

    #train
    hist=model.fit(X_train_r, y_train, epochs=nb_epoch, validation_data=[X_val_r,y_val], batch_size=10,shuffle=True,
                   callbacks=[checkpoint, early_stop],verbose=0)
    y_pre = model.predict(X_val_r)  # y_pre 1200行15列

    my_model = load_model(os.path.join(os.path.join('./model'), 'finetuned_model_test' + str(i) + '.hdf5'))
    loss, accuracy = my_model.evaluate(X_val_r, y_val)

    ten_accuracy.append(accuracy )


print(ten_accuracy)
time_end=time.time()
print('totally cost: ', (time_end-time_start)/60, 'min')

