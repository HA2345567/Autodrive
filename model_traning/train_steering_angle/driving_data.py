import cv2
import numpy as np


xs = []
ys = []

#points to the end of the last batch
train_batch_pointer=0
val_batch_pointer=0

#read data.text
with open("data/driving_dataset/data.txt") as f:
    for line in f:
        xs.append("data/driving_dataset/" + line.split()[0])
        ys.append(float(line.split()[1]) * 3.14159 / 180)  

# get number of images 
num_images = len(xs)

#shuffle list of images and steering angles
c = list(zip(xs, ys))
# np.random.shuffle(c)
xs,ys = zip(*c)

train_xs = xs[0:int(len(xs)*0.8)]
train_ys = ys[0:int(len(xs)*0.8)]

val_xs = xs[int(len(xs)*0.8):]
val_ys = ys[int(len(xs)*0.8):]


num_train_images = len(train_xs)
num_val_images = len(val_xs)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []

    for i in range(batch_size):
        x_out.append(cv2.resize(cv2.imread(train_xs[(train_batch_pointer + i) % num_train_images]) [-150:], (200, 66)) / 255.0)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size 
    return x_out, y_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []

    for i in range(batch_size):
        x_out.append(cv2.resize(cv2.imread(val_xs[(val_batch_pointer + i) % num_val_images]) [-150:], (200, 66)) / 255.0)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size 
    return x_out, y_out






