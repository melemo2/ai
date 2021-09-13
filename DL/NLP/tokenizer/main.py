import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

f = open('C:\\Users\\sonmi\\Desktop\\data\\naver news\\news.txt','r',encoding='utf8')
sentences = f.read()
f.close()

sentences = re.sub('[^w가-힣 ]','',sentences).split(sentences)


tokenizer = Tokenizer(5) # T2S에 쓰이는 단어개수 : 5, oov_token = 'OOV' 적용시 사전에 없는 단어는 인덱스 1 적용
tokenizer.fit_on_texts(sentences)

#print(tokenizer.word_index)
#print(tokenizer.word_counts)

sentences = tokenizer.texts_to_sequences(sentences)
print(sentences)

sentences = pad_sequences(sentences, padding = 'post', maxlen=30) # 패딩
print(sentences)

one_hot = to_categorical(sentences) # one-hot
print(one_hot)