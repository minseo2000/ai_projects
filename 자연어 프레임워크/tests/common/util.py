import numpy as np


# 말뭉치 생성기
def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


# 동시발생 행렬 생성기
def create_to_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx = idx-i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix


# 코사인 유사도 구하기 1이면 두 벡터가 유사, -1이면 두 벡터 관련 x
def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x**2)) + eps)
    ny = y / (np.sqrt(np.sum(y**2)) + eps)

    return np.dot(nx, ny)


if __name__ == '__main__':

    print('preprocess 함수 테스트')

    text = "You say goodbye and I say hello."

    corpus , word_to_id, id_to_word = preprocess(text)

    print(corpus)
    print(word_to_id)
    print(id_to_word)

    print('co-matrix 함수 테스트')

    co_matrix = create_to_matrix(corpus, len(word_to_id))
    print(co_matrix)


    print('cos-similarity 함수 테스트')

    cos_si = cos_similarity(corpus[word_to_id['goodbye']], corpus[word_to_id['i']])

    print(cos_si)


