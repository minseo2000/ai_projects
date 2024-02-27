import numpy as np

# you say goodbye and i say hello.
# 의 동시 발생 행렬
C = np.array([
    [0, 1, 0, 0, 0, 0, 0], # you say goodbye and i say hello . you 근처에는 say가 존재해서 1이 만들어짐.
    [1, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0]
], dtype=np.int32)

print(C[0]) # id가 0인 단어의 벡터 표현



