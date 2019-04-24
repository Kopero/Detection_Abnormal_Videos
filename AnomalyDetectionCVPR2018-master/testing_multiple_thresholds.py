from keras import backend as K
import os
#from importlib import reload

from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
def set_keras_backend(backend):

  if K.backend() != backend:
    os.environ['KERAS_BACKEND'] = backend
    reload(K)
    assert K.backend() == backend

set_keras_backend("theano")

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.regularizers import l2
from keras.optimizers import SGD ,Adagrad
from scipy.io import loadmat, savemat
from keras.models import model_from_json
import theano.tensor as T
import theano
import csv
import configparser
import collections
import time
import csv
import os
from os import listdir
import skimage.transform
from skimage import color
from os.path import isfile, join
import numpy as np
import numpy
from datetime import datetime
from scipy.spatial.distance import cdist,pdist,squareform
import theano.sandbox
import shutil
#theano.sandbox.cuda.use('gpu0')



seed = 7
numpy.random.seed(seed)


def load_model(json_path):  # Function to load the model
    model = model_from_json(open(json_path).read())
    return model

def load_weights(model, weight_path):  # Function to load the model weights
    dict2 = loadmat(weight_path)
    dict = conv_dict(dict2)
    i = 0
    for layer in model.layers:
        weights = dict[str(i)]
        layer.set_weights(weights)
        i += 1
    return model

def conv_dict(dict2):
    i = 0
    dict = {}
    for i in range(len(dict2)):
        if str(i) in dict2:
            if dict2[str(i)].shape == (0, 0):
                dict[str(i)] = dict2[str(i)]
            else:
                weights = dict2[str(i)][0]
                weights2 = []
                for weight in weights:
                    if weight.shape in [(1, x) for x in range(0, 5000)]:
                        weights2.append(weight[0])
                    else:
                        weights2.append(weight)
                dict[str(i)] = weights2
    return dict

# Load Video

def load_dataset_One_Video_Features(Test_Video_Path):

    VideoPath =Test_Video_Path
    f = open(VideoPath, "r")
    words = f.read().split()
    num_feat = int(len(words) / 4096)
    #print('num_feat:', num_feat)
    # Number of features per video to be loaded. In our case num_feat=32, as we divide the video into 32 segments. Note that
    # we have already computed C3D features for the whole video and divided the video features into 32 segments.

    count = -1;
    VideoFeatues = []
    for feat in range(0, num_feat):
        feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])
        count = count + 1
        if count == 0:
            VideoFeatues = feat_row1
        if count > 0:
            VideoFeatues = np.vstack((VideoFeatues, feat_row1))
    AllFeatures = VideoFeatues
    return  AllFeatures



print("Starting testing...")


AllTest_Video_Path = '/home/tienthanh/Downloads/DO_AN_TOT_NGHIEP/testing_data'
#AllTest_Video_Path = "/media/data3/hactienthanh/full_data/training_data1/"
# AllTest_Video_Path contains C3D features (txt file)  of each video. Each file contains 32 features, each of 4096 dimensions.
Results_Path = '/home/tienthanh/Downloads/AnomalyDetectionCVPR2018-master'
#Results_Path = '/media/data3/hactienthanh/code/'
# Results_Path is the folder where you can save your results
Model_dir="/home/tienthanh/Downloads/AnomalyDetectionCVPR2018-master/"
#Model_dir='/content/drive/My Drive/AnomalyDetectionCVPR2018-master/'
# Model_dir is the folder where we have placed our trained weights
#weights_path = Model_dir + 'weightsAnomalyL1L2_5000.mat'
#weights_path ="/media/data3/hactienthanh/code/AnomalyDetectionCVPR2018/Res/weightsAnomalyL1L2_17000.mat"
#weights_path ="/content/drive/My Drive/AnomalyDetectionCVPR2018-master/Res/weightsAnomalyL1L2_20000.mat"
weights_path = "/home/tienthanh/Downloads/AnomalyDetectionCVPR2018-master/weightsAnomalyL1L2_20000.mat"
# weights_path is Trained model weights

model_path = Model_dir + 'model.json'

if not os.path.exists(Results_Path):
       os.makedirs(Results_Path)

All_Test_files= listdir(AllTest_Video_Path)
All_Test_files.sort()

model=load_model(model_path)
load_weights(model, weights_path)
nVideos=len(All_Test_files)
time_before = datetime.now()


'''
for iv in range(nVideos):

    Test_Video_Path = os.path.join(AllTest_Video_Path, All_Test_files[iv])
    video = Test_Video_Path.split('/')[-1]
    if video[0:2] == 'No':
      y_test.append(0)
      n_Normal += 1
    else:
      y_test.append(1)
      n_Abnormal += 1
    inputs=load_dataset_One_Video_Features(Test_Video_Path) # 32 segments features for one testing video
    predictions = model.predict_on_batch(inputs)   # Get anomaly prediction for each of 32 video segments.
    #print('video:', video)
    predicts_videos.append(predictions)
    
    score = max(predictions)
    if score >= 0.5:
      y_pred.append(1)
    else:
      y_pred.append(0)
    #print('{}:{}'.format(video, score))
    if video[0:2] == 'No' and score <= 0.5:
      true_predict += 1
      n_trueNormal += 1
    if video[0:2] != 'No' and score > 0.5:
      true_predict += 1
      n_trueAbnormal += 1
      
    aa=All_Test_files[iv]
    aa=aa[0:-4]
    A_predictions_path = Results_Path + aa + '.mat'  # Save array of 1*32, containing anomaly score for each segment. Please see Evaluate Anomaly Detector to compute  ROC.
    #print(A_predictions_path)
    #print("Total Time took: " + str(datetime.now() - time_before))

print('ACC:', true_predict/nVideos)
print('False alarm rate noraml videos:', 1 - n_trueNormal/n_Normal)
print('False alarm rate abnormal videos:', 1 - n_trueAbnormal/n_Abnormal)


from sklearn import metrics
print(len(y_test))
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# curve ROC
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
plt.figure(figsize=(12,8))
plt.plot(fpr, tpr, label='validation, auc'+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()
'''


ann=open("/home/tienthanh/Downloads/AnomalyDetectionCVPR2018-master/Temporal_Anomaly_Annotation.txt")
line = ann.readline()
line = line[0:-1]
ant = []
while(len(line)>0):
  line = line[0:-1]
  line = line.split()
  ant.append(line)
  #print(line)
  line = ann.readline()
  
ann.close()

count1 = 0
ACC = []

# for draw AUC
y_pre = []
y_true = []

import cv2

root_1 = "/home/tienthanh/Downloads/DO_AN_TOT_NGHIEP/testing_data"
root_2 = "/home/tienthanh/Downloads/testing_videos"

false_alarm_rate_normals = []
false_alarm_rate_abnormals = []
false_fr_normal = []
n_seg_nor = 0.0
n_seg_abnor = 0.0
n_seg_nor_pretrue = 0.0
n_seg_abnor_pretrue = 0.0

# for evaluate use frames
n_fr = 0.0
n_fr_nor = 0.0
n_fr_abnor = 0.0
n_fr_nor_pretrue = 0.0
n_fr_abnor_pretrue = 0.0
y_fr_pre = []
y_fr_true = []


threshold = np.arange(0.5,0.51, 0.05)
for m in threshold:
  num_seg_true = 0.0
  
  for i in range(290):
    n_seg_true_of_video = 0
    path_video = os.path.join(root_2, ant[i][0])
    path_feature = os.path.join(root_1, ant[i][0][:-4] + '_layer_000006.txt')
    #print(path_video)
    #print(path_feature)
    cap = cv2.VideoCapture(path_video)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    #print('num_frames:', num_frames)
    inputs=load_dataset_One_Video_Features(path_feature) # 32 segments features for one testing video
    predict = model.predict_on_batch(inputs)
    print(ant[i])
    print('num_fr:', num_frames)
    cp_predict = np.ones((1,32))
    for k in range(32):
      cp_predict[0][k] = predict[k]
    for k in range(32):
      y_pre.append(cp_predict[0][k])
    
    for k in range(32):
      if predict[k] >= m:
        predict[k] = 1.0
      else:
	predict[k] = 0.0
    
    
    id_fr_of_seg = np.zeros((32,2))
    thirty2_shots = np.round(np.linspace(0, int(num_frames/16.0)-1 , 33))
    #print('thirty2_shots:', thirty2_shots)
    for ishots in range(len(thirty2_shots) - 1):
      ss = int(thirty2_shots[ishots])
      ee = int(thirty2_shots[ishots+1]) - 1
      #print('ss,ee:', ss,ee)
      if ss == ee:
        id_fr_of_seg[ishots][0] = ss*16.0         
        id_fr_of_seg[ishots][1] = (ss+1)*16.0 -1.0
      elif ee < ss:
        id_fr_of_seg[ishots][0] = ss*16.0
        id_fr_of_seg[ishots][1] = (ss+1)*16.0 -1.0 
      else:
        id_fr_of_seg[ishots][0] = ss*16.0
        id_fr_of_seg[ishots][1] = (ee + 1.0)*16.0 - 1.0

    '''
    if ant[i][0][14:17] =='067':    
      for k in range(32):
        print(id_fr_of_seg[k][:], ':', k)    
      print('predict:')
      for k in range(32):
        print(predict[k], ':', k)
    '''       
    ground_truth = np.ones((1, 32))
    if ant[i][2] != '-1':
      if ant[i][4] != '-1':
        for k in range(32):
          if id_fr_of_seg[k][1] < int(ant[i][2]) or  (id_fr_of_seg[k][0] >  int(ant[i][3]) and id_fr_of_seg[k][1] < int(ant[i][4])) or id_fr_of_seg[k][0] > int(ant[i][5]):
            ground_truth[0][k] = 0.0
      else:
        for k in range(32):
          if id_fr_of_seg[k][1] < int(ant[i][2]) or id_fr_of_seg[k][0] > int(ant[i][3]):
            ground_truth[0][k] = 0.0
    else:
      ground_truth = np.zeros((1, 32))

    '''
    if ant[i][0][14:17] =='067':
      print('ground_truth')              
      for k in range(32):
        print(ground_truth[0][k], ':', k)    
    ''' 
   
    
    for k in range(32):
      y_true.append(ground_truth[0][k])
      
       
    

    #=====================================
    # evaluate use frames

    fr_pre = np.zeros((1, int(num_frames)))
    fr_true = np.zeros((1, int(num_frames)))
    if ant[i][2] != '-1':
      if ant[i][4] != '-1':
        fr_true[0][int(ant[i][2]):int(ant[i][3])] = 1.0
        fr_true[0][int(ant[i][4]):int(ant[i][5])] = 1.0
      else:
        fr_true[0][int(ant[i][2]):int(ant[i][3])] = 1.0
    
    for ishots in range(len(thirty2_shots) - 1):
      if predict[ishots] >= m: 
        for t in range(int(id_fr_of_seg[ishots][1]) - int(id_fr_of_seg[ishots][0]) + 1):
          fr_pre[0][int(id_fr_of_seg[ishots][0]) + t] = 1.0
    count_fr = 0.0    
    for k in range(int(num_frames)):
      if fr_true[0][k]==0.0:
        n_fr_nor += 1
        if fr_pre[0][k] == 0.0:
          n_fr_nor_pretrue += 1
    
    
    for k in range(32):
      if ground_truth[0][k] == predict[k]:
        n_seg_true_of_video += 1

    print('n_seg_true_of_video:', n_seg_true_of_video)

    print(len(y_fr_true))
    print(len(y_fr_pre))

    for k in range(int(id_fr_of_seg[31][1]) + 1):
      t = 0.0
      if ant[i][2] != '-1':
        if ant[i][4] != '-1':
          if ( k > int(ant[i][2]) and k < int(ant[i][3]) ) or (k > int(ant[i][4]) and k < int(ant[i][5]) ):
            t = 1.0
        else: 
          if ( k > int(ant[i][2]) and k < int(ant[i][3]) ):
            t = 1.0
      y_fr_true.append(t) 
    
    for ishots in range(32):
      if ishots == 0:
        for l in range(int(id_fr_of_seg[ishots][1]) - int(id_fr_of_seg[ishots][0]) + 1):
          y_fr_pre.append(cp_predict[0][ishots])
      if ishots >= 1 and id_fr_of_seg[ishots-1][0] != id_fr_of_seg[ishots][0]:
        for l in range(int(id_fr_of_seg[ishots][1]) - int(id_fr_of_seg[ishots][0]) + 1):
          y_fr_pre.append(cp_predict[0][ishots])
      
    if len(y_fr_true) != len(y_fr_pre): 
      print(id_fr_of_seg)
      print(id_fr_of_seg[31][1])
      print(len(y_fr_true))
      print(len(y_fr_pre))
      break
    count1 += 1
    if count1 > 4000:
      break 

  
    
  
  
  for i in range(32*290):
    if y_true[i] == 1:
      n_seg_abnor += 1
      if y_pre[i] >= m:
        n_seg_abnor_pretrue += 1
    else:
      n_seg_nor += 1
      if y_pre[i] < m:
        n_seg_nor_pretrue += 1

  
  
  num_seg_true = n_seg_nor_pretrue + n_seg_abnor_pretrue
  print('n_seg_nor_pretrue:', n_seg_nor_pretrue)    
  print('n_seg_abnor_pretrue:', n_seg_abnor_pretrue)  
  print('n_seg_nor:', n_seg_nor)    
  print('n_seg_abnor:', n_seg_abnor) 
  print('num_seg_true:', num_seg_true)
  print('n_fr_nor:', n_fr_nor)
  print('n_fr_nor_pretrue:', n_fr_nor_pretrue)
  
  
  ACC.append(num_seg_true/(32.0*290)*100)
  false_alarm_rate_normals.append(100 - (n_seg_nor_pretrue/n_seg_nor)*100)
  false_alarm_rate_abnormals.append(100 - (n_seg_abnor_pretrue/n_seg_abnor)*100)
  false_fr_normal.append(100 - (n_fr_nor_pretrue/n_fr_nor)*100)
  

  n_seg_nor = 0.0
  n_seg_abnor = 0.0
  n_seg_nor_pretrue = 0.0
  n_seg_abnor_pretrue = 0.0
  
  n_fr = 0.0
  n_fr_nor = 0.0
  n_fr_abnor = 0.0
  n_fr_nor_pretrue = 0.0
  n_fr_abnor_pretrue = 0.0


print('==========================================================')
print(threshold)

print('ACC:', ACC)
print('false_alarm_rate_normals on segs:', false_alarm_rate_normals)
print('false_alarm_rate_abnormals on segs;', false_alarm_rate_abnormals)
print('false_fr_normal:', false_fr_normal)

print('===============')  
print(len(y_fr_true))
print(len(y_fr_pre))

y_true= y_fr_true
y_pre = y_fr_pre

print('===============')  


from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# curve ROC

fpr, tpr, _ = metrics.roc_curve(y_true, y_pre)
auc = metrics.roc_auc_score(y_true, y_pre)
plt.figure(figsize=(12,8))
plt.plot(fpr, tpr, label='validation, auc'+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()














