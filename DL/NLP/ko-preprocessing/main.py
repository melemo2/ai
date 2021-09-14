#!pip install git+https://github.com/haven-jeon/PyKoSpacing.git

import tensorflow as tf
from pykospacing import Spacing
spacing = Spacing()
kospacing_sent = spacing("안녕하세요저는사람입니다.") 

print(kospacing_sent)

from hanspell import spell_checker

sent = "맞춤법 틀리면 외 않되? 쓰고싶은대로쓰면돼지 "
spelled_sent = spell_checker.check(sent)

hanspell_sent = spelled_sent.checked
print(hanspell_sent)