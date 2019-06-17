import pandas as pd
import cv2
import os
import dlib
detector = dlib.get_frontal_face_detector()
def get_faces_df(PATH='./'):
    ''' Returns 3 data frames - for train\validation\testing '''
    
    celeb_data = pd.read_csv(PATH + 'identity_CelebA.txt', sep=" ", header=None)
    celeb_data.columns = ["image", "label"]

    # 0 - train, 1 - validation, 2 - test
    train_val_test = pd.read_csv(PATH+'list_eval_partition.csv', usecols=['partition']).values[:, 0]

    df_train = celeb_data.iloc[train_val_test == 0]
    df_valid = celeb_data.iloc[train_val_test == 1]
    df_test = celeb_data.iloc[train_val_test == 2]

    print('Train images:', len(df_train))
    print('Validation images:', len(df_valid))
    print('Test images:', len(df_test))
    
    return df_train, df_valid, df_test, train_val_test

def get_face_pos(PATH = './'):
    facePos = {}
    bboxes = open(PATH + 'list_bbox_celeba.txt', 'r')

    for bbox in bboxes.readlines()[2: ]:
        bb_info = bbox.split()
        image_file = bb_info[0]
        x_1 = int(bb_info[1])
        y_1 = int(bb_info[2])
        width = int(bb_info[3])
        height = int(bb_info[4])
        facePos[image_file] = [x_1, y_1, width, height]

    return facePos


def getFacePos(image):


    # Get user supplied values
    # Create the haar cascade

    #faceCascade.load(cascPath)
    #image = cv2.imread(imgName)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    else:
        d = faces[0]
        #print(type(d))
        y1 = d.top() if d.top() > 0 else 0
        y2 = d.bottom() if d.bottom() > 0 else 0
        x1 = d.left() if d.left() > 0 else 0
        x2 = d.right() if d.right() > 0 else 0

        return [x1, y1, x2 - x1, y2 - y1]