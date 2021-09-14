import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

f = open('C:\\Users\\sonmi\\Desktop\\data\\naver news\\news.txt','r',encoding='utf8')
sentences = f.read()
f.close()

sentences = re.sub('[^\w가-힣 {0,1}]','',sentences).split(sentences)

