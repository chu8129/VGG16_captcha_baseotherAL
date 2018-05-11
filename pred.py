#!/usr/bin/env python
# -*- coding: utf-8 -*-
# qw @ 2018-05-10 16:31:52

from PIL import Image
import numpy
import random
from keras.models import load_model
from model import decode,read_folder,read_png
from show import plt

model = load_model("model.h5")
characters = open("model.characters").read()

test_folder = "check/"
test_file_list, _characters = read_folder(test_folder)
test_file = random.choice(test_file_list)
x_test, y_test,n_class_test = read_png([test_file], characters)

y_pred = model.predict(x_test)
title = 'real: %s\npred:%s'%(decode(y_test, characters), decode(y_pred, characters))
print title

all_count = 0
right_count = 0
for test_file in test_file_list:
    x_test, y_test,n_class_test = read_png([test_file], characters)
    y_pred = model.predict(x_test)
    y_p =  decode(y_pred, characters)
    y_t = decode(y_test, characters)
    print "real", y_t, "pred", y_p
    all_count += 1
    if y_p == y_t:
        right_count +=1
print "all", all_count, "right", right_count, "percent", 1.0*right_count/all_count

"""
while 1:
    y_pred = y_preds[index]
    y = y_test[index]
    print y
    print y_pred
    print "predict", decode(y_pred, characters)
    #title = 'real: %s\npred:%s'%(decode(y, characters), decode(y_pred, characters))
    #print title
    plt.imshow(x[index], cmap='gray')
    raw_input("go on ?")
"""
