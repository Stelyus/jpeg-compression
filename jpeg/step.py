import numpy as np
from scipy import fftpack
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



# Seems to be the second step
def chroma_subsampling(img_arr,a=2,b=0): 

  def remove_chroma(array, pattern_width):
    if pattern_width == 4:
      return array

    w = array.shape[0]
    pattern_width_dict = {0: w+1, 1: 4, 2: 2}
    step = pattern_width_dict[pattern_width]

    for i in range(0,w):
      if i % step != 0:
        array[i] = [array[i][0], 0, 0]


    # If the pattern is 0 we need to remove the first index 
    if step == pattern_width_dict[0]:
      array[0] = [array[i][0], 0, 0]


  step_width = 2
  
  h = img_arr.shape[0]
  for i in range(0,h):
    th = img_arr[i]
    remove_chroma(th, a) if i % step_width == 0 else remove_chroma(th, b)
    img_arr[i] = th 


def dct_transformation(img_arr):
  def DCT(i, j, img_arr):
    CI = 1/np.sqrt(2) if i == 0 else 1
    CJ = 1/np.sqrt(2) if j == 0 else 1
  
    
    total = 0
    for x in range(0,8):
      min_total = 0
      for y in range(0,8):
        cos = np.cos([((2*x+1)*i*np.pi) / 16,((2*y+1)*j*np.pi) / 16])
        min_total += img_arr[x,y] * cos[0] * cos[1]
      total += min_total

    return np.round(2/8 * CI * CJ * total)



  dct_transformed = np.copy(img_arr)

  if img_arr.shape != (8,8):
    raise ValueError("Img array shape must be 8x8")

  for i in range(0,8):
    for j in range(0,8):
      value_pixel = img_arr[i,j]
      dct_transformed[i, j] = DCT(i, j, img_arr) 

  return dct_transformed

path = "/Users/franckthang/Work/PersonalWork/jpeg-compression/resources/cat.jpg"
img = Image.open(path)
new_image = rgb_to_YCbCr(np.asarray(img))

chroma_subsampling(new_image)


dct_test = np.array([[139,144,149,153,155,155,155,155],
[144,151,153,156,159,156,156,156], [150,155,160,163,158,156,156,156],
[159,161,162,160,160,159,159,159],[159,160,161,162,162,155,155,155],[161,161,161,161,160,157,157,157],[162,162,161,163,162,157,157,157],[162,162,161,161,163,158,158,158]])

dct_transformed = dct_transformation(dct_test)
print(dct_transformed)
#print(fftpack.dct(fftpack.dct(dct_test, axis=0), axis=1)[0,0])


