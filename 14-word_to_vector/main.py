import numpy as np
from w2v_utils import *


def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similariy between u and v

    Arguments:
        u -- a word vector of shape (n,)
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """

    dot = np.dot(u, v)
    norm_u = np.sqrt(np.sum(u**2))
    norm_v = np.sqrt(np.sum(v**2))
    cosine_similarity = dot / (norm_u * norm_v)

    return cosine_similarity


def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task as explained above: a is to b as c is to ____.

    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors.

    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
    """

    # convert words to lower case
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()

    # Get the word embeddings v_a, v_b and v_c
    e_a = word_to_vec_map[word_a]
    e_b = word_to_vec_map[word_b]
    e_c = word_to_vec_map[word_c]

    words = word_to_vec_map.keys()

    max_cosine_sim = -100
    best_word = None

    for w in words:
        if w in [word_a, word_b, word_c]:
            continue

        cosine_sim = cosine_similarity(e_b - e_a, word_to_vec_map[w] - e_c)

        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w

    return best_word


if __name__ == '__main__':
    words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
    triads_to_try = [
        ('italy', 'italian', 'spain'),
        ('india', 'delhi', 'japan'),
        ('man', 'woman', 'boy'),
        ('small', 'smaller', 'large')
    ]
    for triad in triads_to_try:
        print('{} -> {} :: {} -> {}'.format(*triad, complete_analogy(*triad, word_to_vec_map)))
