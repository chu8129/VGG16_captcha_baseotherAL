#!/usr/bin/env python
# -*- coding: utf-8 -*-
# qw @ 2018-05-10 11:36:48


from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random

import string
characters = string.digits + string.ascii_uppercase

width, height, n_len, n_class = 170, 80, 4, len(characters)

generator = ImageCaptcha(width=width, height=height)
random_str = ''.join([random.choice(characters) for j in range(4)])
img = generator.generate_image(random_str)

img.save(random_str+".bmp")
