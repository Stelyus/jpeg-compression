import numpy as np
from PIL import Image

# Seems to be the first step
def rgb_to_YCbCr(img_arr):
  h = img_arr.shape[0]
  w = img_arr.shape[1]


  y_coeff = np.full(img_arr.shape,np.asarray([(77/256),(150/256),(29/256)]))
  cb_coeff = np.full(img_arr.shape,np.asarray([-(44/256),-(87/256),(131/256)]))
  cr_coeff = np.full(img_arr.shape,np.asarray([(131/256),- (110/256),-(21/256)]))
  bias = np.asarray([0,128,128])

  Y_tmp =  np.multiply(y_coeff,img_arr)
  Y = np.sum(Y_tmp, axis=-1).reshape((h,w,1))

  Cb_tmp =  np.multiply(cb_coeff,img_arr)
  Cb = np.sum(Cb_tmp,axis=-1).reshape((h,w,1))

  Cr_tmp =  np.multiply(cr_coeff,img_arr)
  Cr = np.sum(Cr_tmp,axis=-1).reshape((h,w,1))

  new = np.concatenate([Y, Cb, Cr], axis=-1) + bias

  return new 



#def YCbCr_to_rgb():
  #R = Y + 1.402 * (Cr-128)
  #G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
  #B = Y + 1.772 * (Cb - 128)
  #return R, G, B




# Seems to the second step
def chroma_subsampling(img_arr,a=2,b=0): 

  def remove_chroma(array, pattern_width):
    if pattern_width == 4:
      return array

    pattern_width_dict = {0: 1, 1: 4, 2: 2}
    step = pattern_width_dict[pattern_width]
    w = array.shape[0]

    for i in range(0,w):
      if i % step != 0:
        array[i] = [array[i][0], 0, 0]


  step_width = 2
  
  h = img_arr.shape[0]
  for i in range(0,h):
    th = img_arr[i]
    remove_chroma(th, a) if step_width % 2 == 0 else remove_chroma(th, b)
    img_arr[i] = th 




path = "/Users/franckthang/Work/PersonalWork/jpeg-compression/resources/cat.jpg"
img = Image.open(path)
new_image = rgb_to_YCbCr(np.asarray(img))

chroma_subsampling(new_image)
