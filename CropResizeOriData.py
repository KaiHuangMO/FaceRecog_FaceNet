import os
import cv2
import dlib
import copy
detector = dlib.get_frontal_face_detector()
rootdir = 'OriDataset'
targetdir = 'CropDataset'
FACE_DEFAULT_SHAPE = (200, 200)
list = os.listdir(rootdir) #列出文件夹下所有的目录与文件

# 清空文件夹下文件
def clean(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            for f in os.listdir(path_file):
                path_file2 = os.path.join(path_file, f)
                if os.path.isfile(path_file2):
                    os.remove(path_file2)

clean(targetdir)
for i in range(0,len(list)):
    path = os.path.join(rootdir,list[i])
    if os.path.isfile(path) and path.endswith('jpg'):
        img = cv2.imread(path)#你想对文件的操作
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
            copyImg = copy.deepcopy(img)
            cv2.rectangle(copyImg, (pos[0], pos[1]), (pos[0] + pos[2], pos[1] + pos[3]), (0, 255, 0), 2)
            cv2.imshow('reImg', copyImg)
            k = cv2.waitKey(0)
            if k == 27:  # wait for ESC key to exit
                cv2.destroyAllWindows()
            resizeImg = cv2.resize(cropImg, (int(FACE_DEFAULT_SHAPE[0]), int(FACE_DEFAULT_SHAPE[1])))
            targetPath = os.path.join(targetdir,list[i])
            cv2.imwrite(targetPath, resizeImg)