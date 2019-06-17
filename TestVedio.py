import cv2
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
font = cv2.FONT_HERSHEY_SIMPLEX

'''
读入模型和embedding
'''
id_file =  np.load('id_file.npy').item()
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
detector = dlib.get_frontal_face_detector()

# 开启摄像头
# capture frames from a camera with device index=0
cap = cv2.VideoCapture(0)

# loop runs if capturing has been initialized
while (1):

    # reads frame from a camera
    ret, frame = cap.read()
    # 做图像处理
    img = copy.deepcopy(frame)
    img2 = img.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) == 0:
        cv2.imshow('Camera', frame)
    else:
        for d in faces:
            y1 = d.top() if d.top() > 0 else 0
            y2 = d.bottom() if d.bottom() > 0 else 0
            x1 = d.left() if d.left() > 0 else 0
            x2 = d.right() if d.right() > 0 else 0
            pos = [x1, y1, x2 - x1, y2 - y1]
            cropImg = frame[pos[1]:pos[1] + pos[3], pos[0]:pos[0] + pos[2]]
            resizeImg = cv2.resize(cropImg, (int(FACE_DEFAULT_SHAPE[0]), int(FACE_DEFAULT_SHAPE[1])))
            tmpImg = copy.deepcopy(resizeImg)

            imgT = image.img_to_array(resizeImg)
            imgT = preprocess_input(imgT)
            imgT.astype(np.float32)
            pred = infer_model.predict(imgT[None])[0]
            res = -np.matmul(embeddings, pred.T)  # embedding 都一样
            sorted_similar = np.argsort(res, axis=0)
            most_similar = id_file[sorted_similar[0]].split('.')[0]
            img = cv2.putText(img, most_similar, (pos[1], pos[0]), font, 1.2, (255, 255, 255), 2)
            cv2.rectangle(img, (pos[0], pos[1]), (pos[0] + pos[2], pos[1] + pos[3]), (0, 255, 0), 2)

            # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度


    # Display the frame
    cv2.imshow('Camera', img)

    # Wait for 25ms
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the camera from video capture
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()