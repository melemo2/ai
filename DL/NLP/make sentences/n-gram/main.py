import re
f = open('C:\\Users\\sonmi\\Desktop\\data\\naver news\\news.txt','r',encoding='utf8')
sentences = f.read()
f.close()

sentences = re.sub('[^w가-힣 ]','',sentences).split(sentences)
print(list(zip(*[sentences[0].split()[i:] for i in range(3)])))
print(sentences)