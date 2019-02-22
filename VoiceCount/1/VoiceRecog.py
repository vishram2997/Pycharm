from scipy.io.wavfile import read
import numpy as np
import  os


fileDir = os.path.dirname(os.path.realpath('__file__'))
voice =[]
for a in range(10):
    b = read(str(a+1)+ ".wav",'rb')
    voice.append(np.array(b[1],dtype=float))

label = [1,2,3,4,5,6,7,8,9,10]