"""
Approximate time for an augmentation is 20-30 micro seconds a word
"""

from numpy.random import shuffle
from random import choice as rchoice
import string
from random import randrange
from functools import partial
from nla.parallelize import *


def choice(iterable, size):
    """
    faster version of numpy.random.choice function
    :param iterable:
    :param size:
    :return:
    """
    shuffle(iterable)
    return iterable[:size]


def pos_word(word, position):
    """
    split a word at the given position
    :param word: word to split
    :type word: str
    :param position: position to split, 'first', 'middle' and 'end'
    :type position: str
    :return:
    """
    positions = ["first", "middle", "end", "random"]
    assert position in positions, "position argument needs to be either {}".format(
        str(positions)[1:-1]
    )

    marker = len(word) // 2

    if position == "first":
        word_head, word, word_tail = "", word[: min(marker, 4)], word[min(marker, 4) :]

    elif position == "middle":
        word_head, word, word_tail = (
            word[: marker - 2],
            word[marker - 2 : marker + 2],
            word[marker + 2 :],
        )

    elif position == "end":
        word_head, word, word_tail = (
            word[: -min(marker, 4)],
            word[-min(marker, 4) :],
            "",
        )

    else:
        word_head, word, word_tail = "", word, ""

    return word_head, word, word_tail


def swap(word, degree, position="random"):
    """
    swap random characters in a string
    :param word: word to augment
    :type word: str
    :param degree: number of places to swap
    :type degree: int
    :param position: position to augment in every word of the sentence
    :type position: str
    :return:
    """

    word_head, word, word_tail = pos_word(word, position)

    if degree == 1:
        return [
            word_head
            + "".join((word[:i], word[i + 1], word[i], word[i + 2 :]))
            + word_tail
            for i in range(len(word) - 1)
        ]

    word = list(word)
    degree = min(len(word) // 3, degree)

    # idx = choice(range(1, len(word) - 1, 3), size=degree, replace=False)
    start_idx = choice([2, 1], 1)[0]
    idx = choice(list(range(start_idx, len(word) - 1, 3)), size=degree)

    for ix in idx:
        lr = choice([1, -1], 1)[0]
        word[ix], word[ix + lr] = word[ix + lr], word[ix]

    return word_head + "".join(word) + word_tail


def delete(word, degree, position="random"):
    """
    delete random characters from a string
    :param word: word to augment
    :type word: str
    :param degree: number of places to delete
    :type degree: int
    :param position: position to augment in every word of the sentence
    :type position: str
    :return:
    """
    word_head, word, word_tail = pos_word(word, position)

    if degree == 1:
        return [
            word_head + "".join((word[:i], word[i + 1 :])) + word_tail
            for i in range(len(word) - 1)
        ]

    degree = min(len(word) // 2, degree)

    start_idx = choice([0, 1], 1)[0]
    idx = choice(list(range(start_idx, len(word) - 1, 2)), size=degree)
    idx[::-1].sort()

    for ix in idx:
        word = word[:ix] + word[ix + 1 :]

    return word_head + word + word_tail


def insert(word, degree, position="random"):
    """
    insert random characters in a string
    :param word: word to augment
    :type word: str
    :param degree: number of places to insert
    :type degree: int
    :param position: position to augment in every word of the sentence
    :type position: str
    :return:
    """
    word_head, word, word_tail = pos_word(word, position)

    if degree == 1:
        return [
            word_head
            + "".join((word[:i], rchoice(string.ascii_uppercase), word[i:]))
            + word_tail
            for i in range(len(word) - 1)
        ]

    degree = min(len(word), degree)

    start_idx = choice([0, 1], 1)[0]
    idx = choice(list(range(start_idx, len(word) - 1 + degree, 2)), size=degree)
    idx.sort()

    for ix in idx:
        word = word[:ix] + rchoice(string.ascii_uppercase) + word[ix:]

    return word_head + word + word_tail


def replace(word, degree, position="random"):
    """
    replace random characters in a string
    :param word: word to augment
    :type word: str
    :param degree: number of places to replace
    :type degree: int
    :param position: position to augment in every word of the sentence
    :type position: str
    :return:
    """
    word_head, word, word_tail = pos_word(word, position)

    if degree == 1:
        return [
            word_head
            + "".join((word[:i], rchoice(string.ascii_uppercase), word[i + 1 :]))
            + word_tail
            for i in range(len(word) - 1)
        ]

    degree = min(len(word), degree)

    start_idx = choice([0, 1], 1)[0]
    idx = choice(list(range(start_idx, len(word) - 1)), size=degree)

    for ix in idx:
        word = word[:ix] + rchoice(string.ascii_uppercase) + word[ix + 1 :]

    return word_head + word + word_tail


def __fetch__(word, degree, count, method="random", position="random", **kwargs):
    """
    run the given augmentation on a word
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
        "swap": swap,
        "delete": delete,
        "insert": insert,
        "replace": replace,
        "random": [swap, insert, replace, delete],
    }

    assert method in functions, "method argument needs to be either {}".format(
        str(list(functions.keys()))[1:-1]
    )

    words = set()
    iterator = 0

    while len(words) < count and iterator < count * 2:
        function = (
            choice(functions[method], 1)[0] if method == "random" else functions[method]
        )
        a_degree = randrange(0, degree) + 1

        augword = function(word, a_degree, position=position)

        if augword:
            if a_degree == 1:
                augword = choice(augword, 1)[0]

            words.add(augword)
        iterator += 1

    return [(w, word) + tuple(kwargs.values()) for w in words]


# run_parallel wrapper on __fetch__
def fetch(
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
            __fetch__,
            **kwargs,
            degree=degree,
            method=method,
            count=count,
            position=position,
        )
        return run_parallel(words, function)

    else:
        return [
            __fetch__(
                word=word,
                degree=degree,
                count=count,
                method=method,
                position=position,
                **kwargs,
            )
            for word in words
        ]


if __name__ == "__main__":
    query = "DEEP"
    print(
        __fetch__(
            word=query,
            degree=2,
            count=10,
            method="insert",
            position="first",
            dummy_identifier_1="generic1",
            dummy_identifier_2="generic2"
        )
    )

    query = ["DEEP", "NEURAL", "CRAVING"]
    print(
        fetch(
            words=query,
            degree=2,
            count=2,
            method="random",
            position="random",
            parallel=True,
            dummy_identifier_1="generic1",
            dummy_identifier_2="generic2"
        )
    )
