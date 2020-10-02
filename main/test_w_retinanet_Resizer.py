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



import skimage.transform
def Resizer(image, min_side=608, max_side=1024):
    rows, cols, cns = image.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
    rows, cols, cns = image.shape

    pad_w = 32 - rows%32
    pad_h = 32 - cols%32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)


    return new_image


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
for i in tqdm(range(len(path))):
    img_A_path = path[i]
    img_A = utls.imread_color(img_A_path)
    
    # add resize
    img_A = Resizer(img_A)

    img_A = img_A[np.newaxis, :]

    starttime = time.clock()
    out_pred = mbllen.predict(img_A)
    endtime = time.clock()
    print('The ' + str(i+1)+'th image\'s Time:' +str(endtime-starttime)+'s.')
    fake_B = out_pred[0, :, :, :3]
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