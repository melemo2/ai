#!pip install git+https://github.com/haven-jeon/PyKoSpacing.git

import tensorflow as tf
from pykospacing import Spacing
spacing = Spacing()
kospacing_sent = spacing("안녕하세요저는사람입니다.") 

print(kospacing_sent)