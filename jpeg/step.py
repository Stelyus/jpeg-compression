import numpy as np
from PIL import Image

class Jpeg():
  def __init__(self, format=(4,2,0)):
    dct = { (4,4,4): (8,8), (4,2,2): (8,8), (4,2,0): (8,8) }
    
    if format not in dct:
      raise ValueError("Format %s not correct" % format)

    self.chrominance_table = np.array([
      [16, 11, 10, 16, 24, 40, 51, 61],
      [12, 12, 14, 19, 26, 58, 60, 55],
      [14, 13, 16, 24, 40, 57, 69, 56],
      [14, 17, 22, 29, 51, 87, 80, 62],
      [18, 22, 37, 56, 68, 109, 103, 77],
      [24, 36, 55, 64, 81, 104, 113, 92],
      [49, 64, 78, 87, 103, 121, 120, 101],
      [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    self.mcu = dct[format]
    self.format = format

  def _reshaping(self, img_arr):
    h = img_arr.shape[0]
    w = img_arr.shape[1]

    ah = self.mcu[0] - (h % self.mcu[0])
    aw = self.mcu[1] - (w % self.mcu[1])
      
    #print("Module ah %s, aw %s" % (ah, aw))
    #print("Shape image %s" % (img_arr.shape,))
    #print("Last right shape %s" % (img_arr[:,-1].shape,))
    #print("Last end shape %s" % (img_arr[-1].shape,))

    arr_h = np.full((ah, w, 3), img_arr[-1])
    new_arr = np.vstack((img_arr, arr_h))
    arr_w = np.full((aw, h + ah, 3), new_arr[:,-1]).reshape((h + ah,aw,3))
    new_arr = np.hstack((new_arr, arr_w))

  
    print("Before arr shape %s" % (img_arr.shape, ))
    print("New arr shape %s" % (new_arr.shape,))
    return new_arr

  def _rgb_to_YCbCr(self, img_arr):
    # Seems to be the first step
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


  def _chroma_subsampling(self, img_arr):
    # seems to be the second step
    # depending on chroma subsampling, this yields minimum coded unit (mcu) blocks
    # of size 8×8 (4:4:4 – no subsampling), 16×8 (4:2:2), or most commonly 16×16
    # (4:2:0)
    # the encoder must fill the remaining area of the incomplete blocks with some
    # form of dummy data.
    # seems to be the third step

    def remove_chroma(array, pattern_width):
      if pattern_width == 4:
        return array

      w = array.shape[0]
      pattern_width_dict = {0: w+1, 1: 4, 2: 2}
      step = pattern_width_dict[pattern_width]

      for i in range(0,w):
        if i % step != 0:
          array[i] = [array[i][0], 0, 0]


      # if the pattern is 0 we need to remove the first index 
      if step == pattern_width_dict[0]:
        array[0] = [array[i][0], 0, 0]


    step_width = 2
    
    h = img_arr.shape[0]
    for i in range(0,h):
      th = img_arr[i]
      if i % step_width == 0:
        remove_chroma(th, self.format[0])
      else:
        remove_chroma(th, self.format[1])
      img_arr[i] = th 

  def _splitting_blocks(self, img_arr):
    h, w = img_arr.shape[0], img_arr.shape[1]
    partial_split = np.split(img_arr, h / self.mcu[0], axis=0)

    return list(map(lambda x:np.split(x, w / self.mcu[1], axis=1), partial_split))


  def _dct_transformation(self, block):
    def DCT(i, j, block):
      CI = 1/np.sqrt(2) if i == 0 else 1
      CJ = 1/np.sqrt(2) if j == 0 else 1
    
      
      total = 0
      for x in range(0,self.mcu[0]):
        min_total = 0
        for y in range(0,self.mcu[1]):
          cos = np.cos([((2*x+1)*i*np.pi) / 16,((2*y+1)*j*np.pi) / 16])
          min_total += block[x,y] * cos[0] * cos[1]
        total += min_total

      return np.round(2/8 * CI * CJ * total)

    dct_transformed = np.copy(block)

    for i in range(0,self.mcu[0]):
      for j in range(0,self.mcu[1]):
        value_pixel = block[i,j]
        dct_transformed[i, j] = DCT(i, j, block) 

    return dct_transformed

  # Seems to be the fourth step
  def _quantification(self, dct_arr):
    quantified_matrix = np.copy(dct_arr)

    for u in range(0,self.mcu[0]):
      for v in range(0,self.mcu[1]):
        quantified_matrix[u,v] = (dct_arr[u,v] + (self.chrominance_table[u,v] // 2)) // self.chrominance_table[u,v]

    return quantified_matrix


  # Seems to be the fifth step
  def _codage(self, quant_m):
    diag_values = []
    for x in range(1, 9):
      diag = list(zip(range(0,x), range(x-1,-1,-1)))
      if x % 2:
        diag = diag[::-1]
      for pos in diag:
        diag_values.append(quant_m[pos[0], pos[1]])


    for x in range(1,8):
      diag = list(zip(range(x,8), range(7,-1,-1)))
      if x % 2:
        diag = diag[::-1]
      for pos in diag:
        diag_values.append(quant_m[pos[0], pos[1]])


    return diag_values
    #idx_l = len(diag_values) \
            #- next(i for i,v in enumerate(diag_values[::-1]) if v != 0)
    #return diag_values[0:idx_l]
#
  def _compress(self, diag_l):
    # TODO:(Encoding + Huffman but too lazy)
    print(diag_l)
    

  def run(self, imgpath):
    reshaped_image = self._reshaping(np.asarray(Image.open(imgpath)))
    new_image = self._rgb_to_YCbCr(np.asarray(reshaped_image))
    self._chroma_subsampling(new_image)

    blocks = self._splitting_blocks(new_image)

    # TODO: Mean shifting?
    # Mean Shifting: For averaging the image pixels to 0, we mean shift by
    # subtracting 128 from every pixel.

    # Testing purpose only
    block = blocks[5][5][:,:,0]
    dct_transformed = self._dct_transformation(block)
    print(dct_transformed)
    quant_m = self._quantification(block)
    print(quant_m)
    diag_values = self._codage(quant_m)
    self._compress(diag_values)



#def YCbCr_to_rgb():
  #R = Y + 1.402 * (Cr-128)
  #G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
  #B = Y + 1.772 * (Cb - 128)
  #return R, G, B
