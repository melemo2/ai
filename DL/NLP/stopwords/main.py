from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

text=["Family is not an important thing. It's everything.","my english skill is very good"]
sw = stopwords.words("english")
vect = CountVectorizer(stop_words =sw)
print(vect)
print(vect.fit_transform(text).toarray()) 
print(vect.vocabulary_)