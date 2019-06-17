import os
import cv2
import dlib
from tqdm import tqdm
import numpy as np
import os
from keras import backend as K

from keras.models import load_model, Model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
import cv2
import matplotlib.pyplot as plt
#%matplotlib inline
import keras.losses


# 定义loss
def triplet_loss(y_true, y_pred, cosine = True, alpha = 0.2):
    embedding_size = K.int_shape(y_pred)[-1] // 3
    ind = int(embedding_size * 2)
    a_pred = y_pred[:, :embedding_size]
    p_pred = y_pred[:, embedding_size:ind]
    n_pred = y_pred[:, ind:]
    if cosine:
        positive_distance = 1 - K.sum((a_pred * p_pred), axis=-1)
        negative_distance = 1 - K.sum((a_pred * n_pred), axis=-1)
    else:
        positive_distance = K.sqrt(K.sum(K.square(a_pred - p_pred), axis=-1))
        negative_distance = K.sqrt(K.sum(K.square(a_pred - n_pred), axis=-1))
    loss = K.maximum(0.0, positive_distance - negative_distance + alpha)
    return loss

keras.losses.triplet_loss = triplet_loss
model = load_model('m_crop_siamese_xception_best.h5')
inp = model.input[0]
base_model = model.layers[3]
emb_model = model.layers[4]
infer_model = Model(inp, emb_model(base_model(inp)))
from keras.preprocessing import image
FACE_DEFAULT_SHAPE = (200, 200)

rootdir = 'CropDataset'
list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
embeddings = np.zeros((len(list), 128))
count = 0
id_file = {}
for i in range(0,len(list)):
    path = os.path.join(rootdir,list[i])
    if os.path.isfile(path) and path.endswith('jpg'):
        img = cv2.imread(path)
        img = image.img_to_array(img)
        img = preprocess_input(img)
        img.astype(np.float32)
        preds = infer_model.predict(img[None])[0]
        embeddings[i] = preds
        id_file[i] = path
        count += 1
print ('processing ' + str(count) + ' imags')

np.save('m_emb_celebrities.npy', embeddings)
np.save('id_file.npy', id_file)
