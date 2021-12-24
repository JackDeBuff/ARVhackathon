import cv2
import os
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CLAHE:
    def __init__(self, clipLimit=2, tileGridSize=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        
    def applySingle(self, c):
        return self.clahe.apply(c)
    
    def applyRGB(self, img):
        lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        L, a, b = cv2.split(lab_img)

        Lcl = self.clahe.apply(L)

        cl_img = cv2.merge([Lcl, a, b])
        cl_img = cv2.cvtColor(cl_img, cv2.COLOR_LAB2RGB)
        return cl_img
    
    def applyHSV(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        H, S, V = cv2.split(img)
        Hcl = H
        Scl = self.clahe.apply(S)
        Vcl = self.clahe.apply(V)

        cl_img = cv2.merge([Hcl, Scl, Vcl])
        cl_img = cv2.cvtColor(cl_img, cv2.COLOR_HSV2RGB)
        return cl_img
    
    def setCLAHE(self, clipLimit, tileGridSize):
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        
class ContrastStretch:
    def __init__(self, a=0, b=255):
        self.a = a #minimum desired intensity
        self.b = b #maximum desired intensity
        
    def applySingle(self, img):
        c = img.min()
        d = img.max()

        a, b, c, d = int(self.a), int(self.b), int(c), int(d)
        sub = cv2.subtract(img, c)
        scale = cv2.divide(b, cv2.subtract(d, c))
        mul = cv2.multiply(sub, scale)
        result = cv2.add(mul, a)
        return result
    
    def applyRGB(self, img):
        R, G, B = cv2.split(img)

        Rnew = R # we don't want to consider red color, since the color of underwater image is rarly red.
        Gnew = self.applySingle(G)
        Bnew = self.applySingle(B)

        stretch_img = cv2.merge([Rnew, Gnew, Bnew])
        stretch_img = cv2.cvtColor(stretch_img, cv2.COLOR_RGB2HSV)

        H, S, V = cv2.split(stretch_img)

        Hnew = H
        Snew = self.applySingle(S)
        Vnew = self.applySingle(V)

        enhance_img = cv2.merge([Hnew, Snew, Vnew])
        enhance_img = cv2.cvtColor(enhance_img, cv2.COLOR_HSV2RGB)

        return enhance_img