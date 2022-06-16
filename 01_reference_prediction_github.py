# time start
import os
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

time_start = time.time()

import numpy as np
from keras import metrics, Model
from keras.utils import np_utils
from keras.models import load_model
# train with GPU if GPU available
import tensorflow as tf
import keras
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

#强制使用CPU训练
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from config import STRAINS
namelabels = STRAINS
from config import ATCC_GROUPINGS

# Making predictions on the reference dataset
# In this notebook, we'll demonstrate using a trained and fine-tuned CNN to make predictions on an independent test
# set composed of reference isolates. This code illustrates the procedure used to generate the results shown in Figure 2.

## Loading data
# The first step is to load the test dataset (3000 spectra).
t0 = time.time()
X = 'E:\\zhoubo\\data\\nc30\\X_test.npy'
y = 'E:\\zhoubo\\data\\nc30\\y_test.npy'
X = np.load(X)
y = np.load(y)

print(X.shape, y.shape)

nb_class = 30
nb_features = 1000

five_acc,five_anti_acc=list(),list()
for i in range(5):

    # load the finetuned model
    t0 = time.time()
    my_model = load_model('./model/nc30_0.85/finetuned_model_test'+str(i)+'.hdf5')


    # reshape data to the shape that keras likes.
    y_test = np_utils.to_categorical(y, nb_class)
    X_test_r = np.zeros((len(X), nb_features, 1))
    X_test_r[:, :, 0] = X[:, :nb_features]

    # get acc
    loss, accuracy = my_model.evaluate(X_test_r, y_test)
    five_acc.append(accuracy)

    y_pre_init=my_model.predict(X_test_r) #测试子集的预测标签，y_pre(one-hot矩阵，m行15列)代表
    y_true_init=y_test  #测试子集的真实标签(m,15)，y_test是One-hot矩阵，

    y_pre_1 = np.argmax(y_pre_init, axis=1)
    y_true_1 = np.argmax(y_true_init, axis=1)


    # Mapping predictions into antibiotic groupings
    y_ab = np.asarray([ATCC_GROUPINGS[i] for i in y_true_1])
    y_ab_hat = np.asarray([ATCC_GROUPINGS[i] for i in y_pre_1])

    anti_acc = (y_ab_hat == y_ab).mean()
    five_anti_acc.append(anti_acc)

print('five acc:', five_acc)
print('average acc:',np.mean(np.array(five_acc)))
print('five anti acc:',five_anti_acc)
print('five anti acc:',np.mean(np.array(five_anti_acc)))

time_end = time.time()
print('time is ', (time_end-time_start),' s')



