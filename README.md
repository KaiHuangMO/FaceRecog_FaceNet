# 使用FaceNet和Keras进行人脸识别

#### Keras implementation of the paper: [FaceNet: A Unified Embedding for Face Recognition and Clusterin](https://arxiv.org/abs/1503.03832)


* ## Dataset: 
  - **[CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)**
  - Link for downloading face images: [img_align_celeba](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing)

* ## Dependencies
  - **Keras 2 (tensorflow backend)**
  - **open-cv**
  - **tqdm**
  - **pandas**
  
* ## Model
  - Feature extractor model: [MobileNetV2](https://arxiv.org/abs/1801.04381)
  - Embedding model: FaceNet
 
 * ## 运行顺序
  - 训练: Train_Crop_Siamese_Network.py
  
  - 使用：将待匹配人脸放入 OriDataset文件夹中
  - CropResizeOriData.py 进行数据库数据进行人脸查找和resize
  - EmbeddingDataSet.py 将resize后的数据进行embedding
  - TestImg.py 将待匹配人脸放入testImg文件夹下 进行相似人脸输出
  - TestVedio.py 进行实时视频输出