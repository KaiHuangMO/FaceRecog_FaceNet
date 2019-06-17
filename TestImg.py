from tqdm import tqdm
import numpy as np
import os
from keras.models import load_model, Model
from keras.preprocessing import image
#from keras.applications.xception import preprocess_input
from keras.applications.mobilenet_v2 import preprocess_input
import cv2
import dlib
import matplotlib.pyplot as plt
from keras import backend as K
import copy
import keras.losses
id_file =  np.load('id_file.npy').item()
z = 1
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


FACE_DEFAULT_SHAPE = (200, 200)
model = load_model('m_crop_siamese_xception_best.h5')

inp = model.input[0]
base_model = model.layers[3]
emb_model = model.layers[4]

infer_model = Model(inp, emb_model(base_model(inp)))
from keras.preprocessing import image
imgFile = 'CropDataset'
embeddings = np.load('m_emb_celebrities.npy')
testFile = 'testImg'
list = os.listdir(testFile) #列出文件夹下所有的目录与文件
detector = dlib.get_frontal_face_detector()
# 读字典
for i in range(0, len(list)):
    path = os.path.join(testFile, list[i])
    if os.path.isfile(path) and path.endswith('jpg'):
        img = cv2.imread(path)  # 你想对文件的操作
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        if len(faces) == 0:
            continue
        else:
            d = faces[0]
            y1 = d.top() if d.top() > 0 else 0
            y2 = d.bottom() if d.bottom() > 0 else 0
            x1 = d.left() if d.left() > 0 else 0
            x2 = d.right() if d.right() > 0 else 0
            pos = [x1, y1, x2 - x1, y2 - y1]
            cropImg = img[pos[1]:pos[1] + pos[3], pos[0]:pos[0] + pos[2]]
            resizeImg = cv2.resize(cropImg, (int(FACE_DEFAULT_SHAPE[0]), int(FACE_DEFAULT_SHAPE[1])))
            tmpImg = copy.deepcopy(resizeImg)

            img = image.img_to_array(resizeImg)
            img = preprocess_input(img)
            img.astype(np.float32)
            pred = infer_model.predict(img[None])[0]
            res = -np.matmul(embeddings, pred.T)  # embedding 都一样？
            sorted_similar = np.argsort(res, axis=0)
            plt.figure(figsize=(17, 10))
            if not (i + 1):
                plt.subplot(1, 4, 4)
                plt.title('The least similar', fontsize=20)
                plt.subplot(1, 4, 3)
                plt.title('2nd similar', fontsize=20)
                plt.subplot(1, 4, 2)
                plt.title('The most similar', fontsize=20)
                plt.subplot(1, 4, 1)
                plt.title('Original', fontsize=20)
            plt.subplot(1, 4, 1)
            plt.imshow(tmpImg)
            for i, s in enumerate(sorted_similar[:2]):
                print(s)
                img = image.load_img(id_file[s])
                x = image.img_to_array(img)
                plt.subplot(1, 4, i + 2)
                plt.imshow(x.astype('uint8'))
            img = image.load_img(id_file[sorted_similar[-1]])
            x = image.img_to_array(img)
            plt.subplot(1, 4, 4)
            plt.imshow(x.astype('uint8'))
        plt.show()


