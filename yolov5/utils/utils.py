# Author: Zylo117

import math
import os
import uuid
from glob import glob
from typing import Union

import cv2

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'