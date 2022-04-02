"""import sys
import random
import math


def raw_input():
    pass


def main():
    def patgen(k, x):
        # Generates all possible combinations of previous k entries and adds them to the dictionary of dictionaries
        lst = ['0', '1', '2']
        d = ['0', '1', '2']
        for i in range(0, k - 1):
            f = []
            for i in range(0, len(lst)):
                a = lst[i]
                for j in range(0, 3):
                    str = a + d[j]
                    f.append(str)
            lst = f
        # print(lst)
        for key in x:
            # Gets dictionary for each key in x
            for n in range(0, len(lst)):
                # Makes entries for each combination
                x[key][lst[n]] = 0.1

        return x

    k = sys.argv[1]
    k = int(k)  # past k inputs that we need to consider
    c = 0
    ctr = 1
    input_moves = []
    # copy_moves = input_moves[0:len(input_moves)-1] # list of player moves for conditioning
    terminal_output = []
    zero_k = [0, 0, 0]
    prev_zk = [0, 0, 0]
    b = True
    while b == True:
        r = raw_input()
        r = int(r)

        input_moves.append(r)

        copy_moves = input_moves[0:len(input_moves) - 1]

        x = dict()  # Dictinary containing the moves with dictionary of previous moves as values
        x[0] = dict()
        x[1] = dict()
        x[2] = dict()
        if k == 0:
            # When k is zero we just calculate the count for the opponent moves and play the move which can defeat the move with highest count
            zero_k[r] = zero_k[r] + 1

            m = max(prev_zk)

            ind = prev_zk.index(m)

            if ind == 0:
                print(str(1))
            elif ind == 1:
                print(str(2))
            elif ind == 2:
                print(str(1))


        elif k >= len(copy_moves):
            # We play random until we have enough past history to start calculating the probabilities
            luck = random.randint(0, 2)
            terminal_output.append(luck)
            print(str(luck))
        elif k < len(copy_moves):
            x = patgen(k, x)  # Initializes the dictionary x with the right values according to k and sets them to 0.1

            c = math.pow(3, k)
            normalizer = c * 0.1
            for i in range(0, len(copy_moves)):

                if i + k < len(copy_moves):
                    key2 = ''.join(str(x) for x in copy_moves[i:i + k])

                    key1 = copy_moves[i + k]

                    last_val = x[key1][key2]
                    x[key1][key2] = last_val + 1  # This updates the count for every move given past k moves

            for key in x:
                for v in x[key]:
                    if x[key][v] != 0.1:
                        l_val = x[key][v]
                        base = copy_moves[0 + k:].count(key)
                        x[key][v] = l_val / (
                                    base + normalizer)  # This divides the counts to calculate the probabilities, it also normalizes at the same time

            last_k_input = ''.join(
                str(x) for x in copy_moves[len(copy_moves) - k:])  # This builds a string of the past k inputs

            inter_res = []  # This will contain final probabilities before normalizing

            for key in x:
                p_key = round(copy_moves.count(key), 3) / len(copy_moves)
                p_key = round(p_key, 3)

                conditional_key = x[key][last_k_input]  # This takes the conditional probability

                val = conditional_key * p_key

                inter_res.append(val)

            normal_sum = 0
            for i in range(0, len(inter_res)):
                normal_sum = normal_sum + inter_res[i]

            final_res = []
            for j in range(0, len(inter_res)):
                final_res.append(inter_res[j] / normal_sum)

            player_choice = final_res.index(max(final_res))  # Estimate of the players next move

            util_res = []
            for i in range(0, len(final_res)):
                if i == 0:
                    utility = final_res[2] - final_res[1]
                    util_res.append(utility)
                elif i == 1:
                    utility = final_res[0] - final_res[2]
                    util_res.append(utility)
                elif i == 2:
                    utility = final_res[1] - final_res[0]
                    util_res.append(utility)

            our_move = util_res.index(max(util_res))

            print(str(our_move))
            terminal_output.append(r)
            terminal_output.append(our_move)
        prev_zk = zero_k  # Updates to keep track of the previous counts for playing in case of k = 0

        # c = c + 1
        # ctr = ctr + 1
"""
"""
from __future__ import division
from math import sqrt
import random as rnd


def checkGame(a, b):
    if a == '0' and b == '1' or a == '1' and b == '2' or a == '2' and b == '0':
        return -1
    elif a == b:
        return 0
    else:
        return 1


RPS_count = {'000': 3, '001': 3, '002': 3, '010': 3, '011': 3, '012': 3, '020': 3, '021': 3, '022': 3, '100': 3,
             '101': 3, '102': 3, '110': 3, '111': 3, '112': 3, '120': 3, '121': 3, '122': 3, '200': 3, '201': 3,
             '202': 3, '210': 3, '211': 3, '212': 3, '220': 3, '221': 3, '222': 3}

RPS_disp = {'0': 'rock', '1': 'paper', '2': 'scissor'}

wins, ties, losses = 0, 0, 0
last2 = '33'
# T-1, T

# Loops until user presses q
def raw_input(param):
    pass


while (1):
    roll = raw_input('Please type r,p,s, or q\n')

    while (roll not in ['r', 'p', 's', 'q']):
        roll = raw_input("Look: you've got to type r,p,s, or q\n")

    if roll == 'r':
        x = '0'
    elif roll == 'p':
        x = '1'
    elif roll == 's':
        x = '2'
    elif roll == 'q':
        quit()

    if (last2[0] == '3'):
        y = str(rnd.randint(0, 2))
    else:
        r_count = RPS_count[last2 + '0']
        p_count = RPS_count[last2 + '1']
        s_count = RPS_count[last2 + '2']

        tot_count = r_count + p_count + s_count

        q_dist = [r_count / tot_count, p_count / tot_count, 1 - (r_count / tot_count) - (p_count / tot_count)]

        result = [max(q_dist[2] - q_dist[1], 0), max(q_dist[0] - q_dist[2], 0), max(q_dist[1] - q_dist[0], 0)]
        resultnorm = sqrt(result[0] * result[0] + result[1] * result[1] + result[2] * result[2])
        result = [result[0] / resultnorm, result[1] / resultnorm, 1 - result[0] / resultnorm - result[1] / resultnorm]

        y = rnd.uniform(0, 1)

        if y <= result[0]:
            y = '0'
        elif y <= result[0] + result[1]:
            y = '1'
        else:
            y = '2'

        # update dictionary
        RPS_count[last2 + x] += 1

    last2 = last2[1] + x

    print('You played: ' + RPS_disp[x] + '\nI played:   ' + RPS_disp[y] + '\nGAME RESULT (-1 is a loss for you):',
          checkGame(x, y))

    if checkGame(x, y) == -1:
        losses += 1
    elif checkGame(x, y) == 0:
        ties += 1
    elif checkGame(x, y) == 1:
        wins += 1

    print('Wins:', wins, 'Losses:', losses, 'Ties:', ties)"""
"""
import os
import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt
import time

import tensorflow as tf
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
import tensorflow """

import numpy as np
import cv2
import os
import sys

PATH = os.getcwd()+'\\'
cap = cv2.VideoCapture(0)

label = sys.argv[1]

SAVE_PATH = os.path.join(PATH, label)

try:
    os.mkdir(SAVE_PATH)
except FileExistsError:
    pass

ct = int(sys.argv[2])
maxCt = int(sys.argv[3])+1
print("Hit Space to Capture Image")

while True:
    ret, frame = cap.read()
    cv2.imshow('Get Data : '+label,frame[50:350,150:450])
    if cv2.waitKey(1) & 0xFF == ord(' '):
        cv2.imwrite(SAVE_PATH+'\\'+label+'{}.jpg'.format(ct),frame[50:350,150:450])
        print(SAVE_PATH+'\\'+label+'{}.jpg Captured'.format(ct))
        ct+=1
    if ct >= maxCt:
        break

cap.release()
cv2.destroyAllWindows()

"""""""
from keras.models import model_from_json
import numpy as np
from skimage import io
import cv2
import random


def prepImg(pth):
    return cv2.resize(pth, (300, 300)).reshape(1, 300, 300, 3)


def updateScore(play, bplay, p, b):
    winRule = {'rock': 'scissor', 'scissor': 'paper', 'paper': 'rock'}
    if play == bplay:
        return p, b
    elif bplay == winRule[play]:
        return p + 1, b
    else:
        return p, b + 1


with open('model.json', 'r') as f:
    loaded_model_json = f.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("modelweights.h5")
print("Loaded model from disk")

shape_to_label = {'rock': np.array([1., 0., 0.]), 'paper': np.array([0., 1., 0.]), 'scissor': np.array([0., 0., 1.])}
arr_to_shape = {np.argmax(shape_to_label[x]): x for x in shape_to_label.keys()}

options = ['rock', 'paper', 'scissor']
winRule = {'rock': 'scissor', 'scissor': 'paper', 'paper': 'rock'}
rounds = 0
botScore = 0
playerScore = 0

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
loaded_model.predict(prepImg(frame[50:350, 100:400]))

NUM_ROUNDS = 3
bplay = ""

while True:
    ret, frame = cap.read()
    frame = frame = cv2.putText(frame, "Press Space to start", (160, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0),
                                2, cv2.LINE_AA)
    cv2.imshow('Rock Paper Scissor', frame)
    if cv2.waitKey(1) & 0xff == ord(' '):
        break

for rounds in range(NUM_ROUNDS):
    pred = ""
    for i in range(90):
        ret, frame = cap.read()

        # Countdown
        if i // 20 < 3:
            frame = cv2.putText(frame, str(i // 20 + 1), (320, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (250, 250, 0), 2,
                                cv2.LINE_AA)

        # Prediction
        elif i / 20 < 3.5:
            pred = arr_to_shape[np.argmax(loaded_model.predict(prepImg(frame[50:350, 100:400])))]

        # Get Bots Move
        elif i / 20 == 3.5:
            bplay = random.choice(options)
            print(pred, bplay)

        # Update Score
        elif i // 20 == 4:
            playerScore, botScore = updateScore(pred, bplay, playerScore, botScore)
            break

        cv2.rectangle(frame, (100, 150), (300, 350), (255, 255, 255), 2)
        frame = cv2.putText(frame, "Player : {}      Bot : {}".format(playerScore, botScore), (120, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, pred, (150, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, "Bot Played : {}".format(bplay), (300, 140), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (250, 250, 0), 2, cv2.LINE_AA)
        cv2.imshow('Rock Paper Scissor', frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

if playerScore > botScore:
    winner = "You Won!!"
elif playerScore == botScore:
    winner = "Its a Tie"
else:
    winner = "Bot Won.."

while True:
    ret, frame = cap.read()
    frame = cv2.putText(frame, winner, (230, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, "Press q to quit", (190, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2,
                        cv2.LINE_AA)
    frame = cv2.putText(frame, "Player : {}      Bot : {}".format(playerScore, botScore), (120, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
    cv2.imshow('Rock Paper Scissor', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# train
# -*- coding: utf-8 -*-
"""Rock Paper Scissor GitHub
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1bC3cfmlRCUSjd4OX1CHpuVvIeO6bE8_I
"""

import tensorflow.compat.v2 as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io
import os
import pickle
import sys

from google.colab import drive
drive.mount('/content/drive')

"""Preparing our Data"""

DATA_PATH = sys.argv[1] # Path to folder containing data

shape_to_label = {'rock':np.array([1.,0.,0.,0.]),'paper':np.array([0.,1.,0.,0.]),'scissor':np.array([0.,0.,1.,0.]),'ok':np.array([0.,0.,0.,1.])}
arr_to_shape = {np.argmax(shape_to_label[x]):x for x in shape_to_label.keys()}

imgData = list()
labels = list()

for dr in os.listdir(DATA_PATH):
    if dr not in ['rock','paper','scissor']:
        continue
    print(dr)
    lb = shape_to_label[dr]
    i = 0
    for pic in os.listdir(os.path.join(DATA_PATH,dr)):
        path = os.path.join(DATA_PATH,dr+'/'+pic)
        img = cv2.imread(path)
        imgData.append([img,lb])
        imgData.append([cv2.flip(img, 1),lb]) #horizontally flipped image
        imgData.append([cv2.resize(img[50:250,50:250],(300,300)),lb]) # zoom : crop in and resize
        i+=3
    print(i)

np.random.shuffle(imgData)

imgData,labels = zip(*imgData)

imgData = np.array(imgData)
labels = np.array(labels)

"""Model"""

from keras.models import Sequential,load_model
from keras.layers import Dense,MaxPool2D,Dropout,Flatten,Conv2D,GlobalAveragePooling2D,Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.applications.densenet import DenseNet121

"""DenseNet"""

densenet = DenseNet121(include_top=False, weights='imagenet', classes=3,input_shape=(300,300,3))
densenet.trainable=True

def genericModel(base):
    model = Sequential()
    model.add(base)
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(3,activation='softmax'))
    model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['acc'])
    return model

dnet = genericModel(densenet)

checkpoint = ModelCheckpoint(
    'model.h5',
    monitor='val_acc',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode='auto'
)

es = EarlyStopping(patience = 3)

history = dnet.fit(
    x=imgData,
    y=labels,
    batch_size = 16,
    epochs=8,
    callbacks=[checkpoint,es],
    validation_split=0.2
)

dnet.save_weights('model.h5')

with open("model.json", "w") as json_file:
    json_file.write(dnet.to_json())
