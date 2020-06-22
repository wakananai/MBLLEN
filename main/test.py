from glob import glob
import numpy as np
import scipy
import keras
import os
import Network
import utls
import time
import cv2
import argparse
from tqdm import tqdm

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", type=str, default='../input', help='test image folder')
parser.add_argument("--result", "-r", type=str, default='../result', help='result folder')
parser.add_argument("--model", "-m", type=str, default='Syn_img_lowlight_withnoise', help='model name')
parser.add_argument("--com", "-c", type=int, default=0, help='output with/without origional image and mid result')
parser.add_argument("--highpercent", "-hp", type=int, default=95, help='should be in [85,100], linear amplification')
parser.add_argument("--lowpercent", "-lp", type=int, default=5, help='should be in [0,15], rescale the range [p%,1] to [0, 1]')
parser.add_argument("--gamma", "-g", type=int, default=8, help='should be in [6,10], increase the saturability')
parser.add_argument("--maxrange", "-mr", type=int, default=8, help='linear amplification range')
arg = parser.parse_args()

result_folder = arg.result
if not os.path.isdir(result_folder):
    os.makedirs(result_folder)

input_folder = arg.input
path = glob(input_folder+'/*.*')

model_name = arg.model
mbllen = Network.build_mbllen((None, None, 3))
mbllen.load_weights('../models/'+model_name+'.h5')
opt = keras.optimizers.Adam(lr=2 * 1e-04, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
mbllen.compile(loss='mse', optimizer=opt)

flag = arg.com
lowpercent = arg.lowpercent
highpercent = arg.highpercent
maxrange = arg.maxrange/10.
hsvgamma = arg.gamma/10.

crop_size=512
min_crop_size = 32

for i in tqdm(range(len(path))):
    img_A_path = path[i]
    img_A = utls.imread_color(img_A_path)
    img_h, img_w, _ = img_A.shape
    

    starttime = time.clock()
    
    shift_X = shift_Y = crop_size
    out_plane = np.zeros((img_h, img_w, 3), dtype='float32')
    
    for x in range(0, img_w, shift_X):
        for y in range(0, img_h, shift_Y):    
                X_upper = min(x + shift_X, img_w)
                Y_upper = min(y + shift_Y, img_h)
                X_lower = max(0, X_upper-shift_X)
                Y_lower = max(0, Y_upper-shift_Y)    
                input_img = np.zeros((crop_size, crop_size,3))
                size_Y = Y_upper - Y_lower
                size_X = X_upper - X_lower
                
                input_img[:size_Y,:size_X,:] = img_A[Y_lower:Y_upper, X_lower:X_upper, :]
                input_img = input_img[np.newaxis, :]
                
                out_pred = mbllen.predict(input_img)

                out_plane[Y_lower:Y_upper, X_lower:X_upper, :] = out_pred[0, :size_Y, :size_X, :3]
                endtime = time.clock()
                # print('The ' + str(i+1)+'th image\'s Time:' +str(endtime-starttime)+'s.')
    
    
    # fake_B = out_pred[0, :, :, :3]
    fake_B = out_plane
    fake_B_o = fake_B

    gray_fake_B = fake_B[:, :, 0] * 0.299 + fake_B[:, :, 1] * 0.587 + fake_B[:, :, 1] * 0.114
    percent_max = sum(sum(gray_fake_B >= maxrange))/sum(sum(gray_fake_B <= 1.0))
    # print(percent_max)
    max_value = np.percentile(gray_fake_B[:], highpercent)
    if percent_max < (100-highpercent)/100.:
        scale = maxrange / max_value
        fake_B = fake_B * scale
        fake_B = np.minimum(fake_B, 1.0)

    gray_fake_B = fake_B[:,:,0]*0.299 + fake_B[:,:,1]*0.587 + fake_B[:,:,1]*0.114
    sub_value = np.percentile(gray_fake_B[:], lowpercent)
    fake_B = (fake_B - sub_value)*(1./(1-sub_value))

    imgHSV = cv2.cvtColor(fake_B, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(imgHSV)
    S = np.power(S, hsvgamma)
    imgHSV = cv2.merge([H, S, V])
    fake_B = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2RGB)
    fake_B = np.minimum(fake_B, 1.0)

    if flag:
        outputs = np.concatenate([img_A[0,:,:,:], fake_B_o, fake_B], axis=1)
    else:
        outputs = fake_B

    filename = os.path.basename(path[i])
    img_name = result_folder+'/' + filename
    # scipy.misc.toimage(outputs * 255, high=255, low=0, cmin=0, cmax=255).save(img_name)
    outputs = np.minimum(outputs, 1.0)
    outputs = np.maximum(outputs, 0.0)
    utls.imwrite(img_name, outputs)