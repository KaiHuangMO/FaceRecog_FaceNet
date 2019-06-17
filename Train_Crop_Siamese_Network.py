from utils import get_faces_df, get_face_pos, getFacePos
import numpy as np
from keras.utils import Sequence, multi_gpu_model
from keras.layers import Input, Dense, LeakyReLU, Concatenate, Lambda, BatchNormalization, GlobalAveragePooling2D
#from keras.applications.xception import Xception, preprocess_input
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input

from keras import backend as K
from keras.models import Model
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import cv2

import matplotlib.pyplot as plt
#%matplotlib inline

BATCH_SIZE = 2
FACE_DEFAULT_SHAPE = (100, 100)
PARALLEL = True
facePos = get_face_pos('../../data/Anno/')  # 删除了前两行文件
# Path to celeb dataset
#PATH = '/workspace/dataset/'

df_train, df_valid, df_test, _ = get_faces_df()

PATH = '../../data/'  # change

files = df_train.iloc[df_train['label'].values == 27]['image'].values
rnd_files = np.random.choice(files, 8, replace=False)
file = rnd_files[0]
pos = [46, 86, 89, 90]
'''
img = plt.imread(PATH + 'img_align_celeba/{}'.format(file))
plt.imshow(img)
pos = getFacePos(img)
print (pos)

plt.gca().add_patch(plt.Rectangle(xy=(pos[0], pos[1]),width=pos[2],height=pos[3],fill=False, linewidth=2))
plt.show()


#plt.figure(figsize=(12,8))
for i, file in enumerate(rnd_files):
    #plt.subplot(2,4,i+1)
    #img = plt.imread(PATH + 'img_align_celeba/{}'.format(file))
    #plt.imshow(img)
    img = cv2.imread(PATH + 'img_align_celeba/{}'.format(file))
    pos = getFacePos(img)
    print(pos)
    pos = [46, 86, 89, 90]
    #cv2.rectangle(img, (pos[0], pos[1]), (pos[0] + pos[2], pos[1] + pos[3]), (0, 255, 0), 2)
    cropImg = img[pos[1]:pos[1] + pos[3], pos[0]:pos[0] + pos[2]]
    resizeImg = cv2.resize(cropImg, (int(200), int(200)))
    cv2.imshow('image', resizeImg)
    img = image.img_to_array(resizeImg)

    k = cv2.waitKey(0)
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()
    #plt.show()
z = 1
exit()
'''
# Create base model (convolution features extractor)
PARALLEL = False  # I dont have multi GPU
# xception = Xception(include_top=False, weights=None, input_shape = FACE_DEFAULT_SHAPE + (3,))

xception = MobileNetV2(include_top=False, weights=None, input_shape=FACE_DEFAULT_SHAPE + (3,))
# xception = mobilenet

output = GlobalAveragePooling2D()(xception.output)
base_model = Model(xception.input, output)


def embedder(conv_feat_size):
    '''
    Takes the output of the conv feature extractor and yields the embeddings
    '''
    input = Input((conv_feat_size,), name='input')
    normalize = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='normalize')
    x = Dense(512)(input)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(128)(x)
    x = normalize(x)
    model = Model(input, x)
    return model


def get_siamese_model(base_model):
    inp_shape = K.int_shape(base_model.input)[1:]
    conv_feat_size = K.int_shape(base_model.output)[-1]

    input_a = Input(inp_shape, name='anchor')
    input_p = Input(inp_shape, name='positive')
    input_n = Input(inp_shape, name='negative')
    emb_model = embedder(conv_feat_size)
    output_a = emb_model(base_model(input_a))
    output_p = emb_model(base_model(input_p))
    output_n = emb_model(base_model(input_n))

    merged_vector = Concatenate(axis=-1)([output_a, output_p, output_n])
    model = Model(inputs=[input_a, input_p, input_n],
                  outputs=merged_vector)

    return model


model = get_siamese_model(base_model)
# model.load_weights('siamese_xception.h5')
if PARALLEL:
    parallel_model = multi_gpu_model(model, 2)

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


class TripletImageLoader(Sequence):
    def __init__(self, df, preprocess_function, img_shape, facePos, batchSize=16, flip=False):
        self.files = df['image'].values
        self.batchSize = batchSize
        self.y = df['label'].values
        self.N = len(self.y)
        self.facePos = facePos
        self.shape = img_shape
        self.function = preprocess_function  #归一化处理
        self.flip = flip

    def load_image(self, file):
        #img = image.load_img(PATH + 'img_align_celeba/{}'.format(file))
        img = cv2.imread(PATH + 'img_align_celeba/{}'.format(file))
        #img = image.  # 这里需要改
        cropImg = img[pos[1]:pos[1] + pos[3], pos[0]:pos[0] + pos[2]]
        resizeImg = cv2.resize(cropImg, (int(self.shape[0]), int(self.shape[1])))

        img = image.img_to_array(resizeImg)
        img = self.function(img)
        if self.flip:
            if np.random.randint(0, 2):  # do flippings in 50% of the time
                img = img[:, ::-1, :]
        return img

    # gets the number of batches this generator returns
    def __len__(self):
        l, rem = divmod(self.N, self.batchSize)
        return (l + (1 if rem > 0 else 0))

    # shuffles data on epoch end
    def on_epoch_end(self):
        a = np.arange(len(self.y))
        np.random.shuffle(a)
        self.files = self.files[a]
        self.y = self.y[a]

    # gets a batch with index = i
    def __getitem__(self, i):
        start = i * self.batchSize
        stop = np.min([(i + 1) * self.batchSize, self.N])  # clip stop index to be <= N
        # Memory preallocation
        ANCHOR = np.zeros((stop - start,) + self.shape + (3,))
        POSITIVE = np.zeros((stop - start,) + self.shape + (3,))
        NEGATIVE = np.zeros((stop - start,) + self.shape + (3,))
        ancor_labels = self.y[start:stop]
        ancor_images = self.files[start:stop]
        pos_images = []
        neg_images = []
        count = 0
        for k, label in enumerate(ancor_labels):
            pos_idx = np.where(self.y == label)[0]
            neg_idx = np.where(self.y != label)[0]
            neg_images.append(self.files[np.random.choice(neg_idx)])
            pos_idx_hat = pos_idx[(pos_idx < start) | (pos_idx > stop)]
            if len(pos_idx_hat):
                pos_images.append(self.files[np.random.choice(pos_idx_hat)])
            else:
                # positive examples are within the batch or just 1 example in dataset
                pos_images.append(self.files[np.random.choice(pos_idx)])
            count += 1
            if count % 1000 == 0:
                print(count)

        for k, (a, p, n) in enumerate(zip(ancor_images, pos_images, neg_images)):
            ANCHOR[k] = self.load_image(a)
            POSITIVE[k] = self.load_image(p)
            NEGATIVE[k] = self.load_image(n)
        print(count)
        return [ANCHOR, POSITIVE, NEGATIVE], np.empty(
            k + 1)  # we don't need labels so we reutrn dummy label (Keras requierments)

train_gen = TripletImageLoader(df_train, preprocess_input, FACE_DEFAULT_SHAPE, facePos, batchSize = BATCH_SIZE)
print ('train_gen done')
valid_gen = TripletImageLoader(df_valid, preprocess_input, FACE_DEFAULT_SHAPE, facePos, batchSize = BATCH_SIZE)
print ('valid_gen done')
import os
# 使用第一张与第三张GPU卡
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if PARALLEL:
    parallel_model.compile(Adam(lr = 0.0001), loss = triplet_loss)
else:
    model.compile(Adam(lr = 0.0001), loss = triplet_loss)

checkpoint = ModelCheckpoint('crop_siamese_xception.h5', monitor='val_loss',
                             verbose=1, save_best_only=True, save_weights_only=True)

if PARALLEL:
    parallel_model.fit_generator(train_gen, steps_per_epoch=len(train_gen),
                                 epochs=3, validation_data=valid_gen, validation_steps=len(valid_gen),
                                 workers=12, use_multiprocessing=True, callbacks=[checkpoint])
else:
    # Change workers>1 and use_multiprocessing=True if you're working on Linux
    #model.fit_generator(train_gen, steps_per_epoch=len(train_gen),
    #                    epochs=2, validation_data=valid_gen, validation_steps=len(valid_gen),
    #                    workers=12, use_multiprocessing=True, callbacks=[checkpoint])
    model.fit_generator(train_gen, steps_per_epoch=len(train_gen),
                        epochs=20, validation_data=valid_gen, validation_steps=len(valid_gen),callbacks=[checkpoint])
    # Load best model
    #model.load_weights('siamese_xception.h5')

# And save the whole model
model.save('crop_siamese_xception.h5', include_optimizer=False)