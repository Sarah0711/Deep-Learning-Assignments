import numpy as np
import os
import cv2
import argparse
from glob import glob
from sys import argv
def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.4
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      # noisy = image + gauss
      noisy = gauss
      return noisy
   elif noise_typ == "sp":
      row,col,ch = image.shape
      s_vs_p = 0.05
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)
      noisy = image + image * gauss
      return noisy


# parser = argparse.ArgumentParser(description="Parser for Fast-Neural-Style")
# parser.add_argument("--image_path", type=str, required=True, help="path to image dataset")
# args = parser.parse_args()

for filename in glob("*5.JPG"):
   print(filename)
   im = cv2.imread(filename, 1)
   im = im / 255.
   im = noisy("gauss", im)
   #im = noisy("sp", im)
   im = im * 255
   cv2.imwrite(filename + "justnoise_.png", im)



