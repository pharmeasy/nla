import pickle
from nla.keyboard.randaug import choice, pos_word
from random import randrange
import re
from functools import partial
from nla.parallelize import *


path = "/Users/pharmeasy/git/dataset/augmentation/nla/keyboard.pkl"
nnkey = pickle.load(open(path, "rb"))


# operations at a given index
def nn_insert(word, degree, position="random"):
    """
    insert the character at the index in the word
    :param word: word to augment
    :type word: str
    :param degree: number of places to insert
    :type degree: int
    :param position: position to augment in every word of the sentence
    :type position: str
    """
    word_head, word, word_tail = pos_word(word, position)

    degree = min(len(word), degree)

    start_idx = choice([0, 1], 1)[0]
    idx = choice(list(range(start_idx, len(word) - 1 + degree, 2)), size=degree)
    idx.sort()

    for ix in idx:
        word = word[:ix] + choice(nnkey[word[ix]], 1)[0] + word[ix:]

    return word_head + word + word_tail, idx


def nn_replace(word, degree, position="random"):
    """
    replace with the character at the index in the word
    :param word: word to augment
    :type word: str
    :param degree:
    :type degree:
    :param position: position to augment in every word of the sentence
    :type position: str
    :return:
    """
    word_head, word, word_tail = pos_word(word, position)

    degree = min(len(word), degree)

    start_idx = choice([0, 1], 1)[0]
    idx = choice(list(range(start_idx, len(word) - 1)), size=degree)

    for ix in idx:
        word = word[:ix] + choice(nnkey[word[ix]], 1)[0] + word[ix + 1 :]

    return word_head + word + word_tail, idx


def nn_swap(word, degree, position="random"):
    """
    replace with the character at the index in the word
    :param word: word to augment
    :type word: str
    :param degree:
    :type degree:
    :param position: position to augment in every word of the sentence
    :type position: str
    :return:
    """
    word_head, word, word_tail = pos_word(word, position)

    word, idx = choice([nn_insert, nn_replace], 1)[0](word, degree)
    word = list(word)

    for ix in idx:
        if ix > 0:
            lr = choice([1, -1, 0], 1)[0]
            word[ix], word[ix + lr] = word[ix + lr], word[ix]

    return word_head + "".join(word) + word_tail, idx


def __nn_fetch__(word, degree, count, method="random", position="random", **kwargs):
    """
    run the given augmentation
    :param word: word to augment
    :type word: str
    :param degree: number of places to augment in the string
    :type degree: int
    :param count: number of outputs
    :type count: int
    :param method: method of augmentation
    :type method: str
    :param position: position to augment in every word of the sentence
    :type position: str
    :return:
    """
    functions = {
        "swap": nn_swap,
        "insert": nn_insert,
        "replace": nn_replace,
        "random": [nn_swap, nn_insert, nn_replace, nn_replace, nn_replace],
    }

    assert method in functions, "method argument needs to be either {}".format(
        str(list(functions.keys()))[1:-1]
    )

    word_head, word, word_tail = pos_word(word, position)

    if re.match(r"[^A-Z]", word):
        return [(str(word), str(word))]

    words = set()
    iterator = 0

    while len(words) < count and iterator < count * 2:
        # select and run the function based on the method
        function = (
            choice(functions[method], 1)[0] if method == "random" else functions[method]
        )

        # select a degree based on the degree argument
        a_degree = randrange(0, degree) + 1

        # call the selected function
        augword = function(word, a_degree)[0]
        words.add(word_head + augword + word_tail)

        iterator += 1

    word = word_head + word + word_tail
    return [(w, word) + tuple(kwargs.values()) for w in words]


# run_parallel wrapper on __nn_fetch__
def nn_fetch(
    words, degree, count, method="random", position="random", parallel=True, **kwargs
):
    """
    run the given augmentation on given list of words
    :param words: list of words to augment
    :type words: list
    :param degree: number of places to augment in the string
    :type degree: int
    :param count: number of outputs
    :type count: int
    :param method: method of augmentation
    :type method: str
    :param position: position to augment in every word of the sentence
    :type position: str
    :param parallel: run in parallel
    :type parallel: bool
    :param kwargs:
    :return:
    """

    if parallel:
        function = partial(
            __nn_fetch__,
            **kwargs,
            degree=degree,
            method=method,
            count=count,
            position=position,
        )
        return run_parallel(words, function)

    else:
        return [
            __nn_fetch__(
                word, degree, count, method=method, position=position, **kwargs
            )
            for word in words
        ]


if __name__ == "__main__":
    query = "DEEP"
    print(
        __nn_fetch__(
            word=query,
            degree=1,
            count=10,
            method="replace",
            position="first",
            identifier="generic",
        )
    )

    query = ["DEEP", "NEURAL", "CRAVING"]
    print(
        nn_fetch(
            words=query,
            degree=2,
            count=10,
            method="insert",
            position="first",
            parallel=True,
            identifier="generic",
        )
    )
