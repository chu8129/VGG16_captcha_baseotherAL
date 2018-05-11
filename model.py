#!/usr/bin/env python
# -*- coding: utf-8 -*-
# qw @ 2018-05-10 11:46:38

import logging
logging.basicConfig(level=logging.INFO)
from PIL import Image
from keras.models import *
from keras.layers import *


def read_folder(folder_path):
    train_file = []
    characters = set()
    for file_name in os.listdir(folder_path):
        path = os.path.join(folder_path, file_name)
        if os.path.isfile(path):
            name = file_name.split(".")[0]
            for _c in name:
                characters.add(_c)
            train_file.append([path, name])
    return train_file, "".join(characters)


def read_png(train_file_list, characters, height=80, width=170, character_num=4):
    batch_size = len(train_file_list)
    n_class = len(characters)
    logging.info("sample:%s, class:%s"%(batch_size, n_class))
    x = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    logging.info(tuple(x.shape))
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(character_num)]
    logging.info(y[0].shape)

    characters = characters
    logging.info("read file count:%s"%len(train_file_list))
    for _index in xrange(len(train_file_list)):
        logging.debug("read index:%s"%_index)
        file_path, file_name = train_file_list[_index]
        logging.debug("file:%s, code:%s"%(file_path, file_name))
        img = Image.open(file_path).convert("RGB")
        x[_index] = img 
        for j, ch in enumerate(file_name[:character_num]):
            logging.debug("index:%s, num_classified:%s y:%s"%(_index, j, characters.find(ch)))
            y[j][_index, :] = 0
            y[j][_index, characters.find(ch)] = 1
    return x,y,n_class


def decode(y, characters):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])



def start(): 
    height = 80
    width = 170
    
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    
    train_folder = "train/"
    test_folder = "test/"
    

    train_file_list, characters = read_folder(train_folder) 
    for line in  train_file_list[:10]:print line
    print characters
    x_train, y_train, n_class = read_png(train_file_list, characters)
    print x_train[0][40], len(x_train[0][40])
    print x_train[1][40], len(x_train[1][40])
    print x_train[0][40] == x_train[1][40]

    test_file_list, _characters = read_folder(test_folder)
    x_test, y_test,n_class_test = read_png(test_file_list, characters)
    
    input_tensor = Input(shape=(height, width, 3))
    x = input_tensor
    
    for i in range(4):
        print i,x
        x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
        x = Conv2D(32*2**i, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
    logging.info("add conv2d maxpooling2D done")

    x = Flatten()(x)
    logging.info("add flatten done")

    x = Dropout(0.25)(x)
    logging.info("add dropout done")

    x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]
    logging.info("make last classfication done")

    model = Model(input=input_tensor, output=x)
    logging.info("combine model done")
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    logging.info("compile done")
    
    model.fit(
        x_train, 
        y_train, 
        validation_data=(x_test, y_test),
        batch_size=1000,
        epochs=30,
        verbose=1,
        validation_split=0,
        shuffle=True,
        callbacks=[],
        class_weight=None,
        sample_weight=None)
    
    model.save("model.h5")
    with open("model.characters", "w") as fw:
        fw.write(characters)

if __name__ == "__main__":
    start()
