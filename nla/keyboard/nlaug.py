from functools import partial
from nla.keyboard.randaug import *
from nla.keyboard.keyaug import *
from nla.parallelize import *


def __rand_sent_aug__(
    sentence, degree, count, method="random", position="random", **kwargs
):
    """
    run the given random augmentation on a sentence
    :param sentence: sentence to augment
    :type sentence: str
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
    words = []

    # functions to select from
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

    # running for desired count
    for _ in range(count):
        # select the function based on the method
        function = (
            choice(functions[method], 1)[0] if method == "random" else functions[method]
        )

        # # select a degree based on the degree argument
        # a_degree = randrange(0, degree) + 1

        # call the selected function
        augword = function(word=sentence, degree=degree, position=position)

        if augword:
            if degree == 1:
                augword = choice(augword, 1)[0]

            # append the augmented word with other identifiers
            words.append([augword, sentence] + list(kwargs.values()))
    return choice(words, size=count)


def __keyboard_sent_aug__(
    sentence, degree, count, method="random", position="random", **kwargs
):
    """
    run the given augmentation on a sentence
    :param sentence: sentence to augment
    :type sentence: str
    :param degree: number of places to augment in the string
    :type degree: int
    :param method: method to augment
    :type method: str
    :param count: number of outputs
    :type count: int
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

    all_augword = []

    # running for desired count
    for _ in range(count):
        augword = []

        # running for every word in the sentence
        for w in sentence.split():
            # return the query as it is if it doesn't contain only alphabets
            if re.findall(r"[^A-Z]", w):
                augword.append(str(w))
                continue

            # select and run the function based on the method
            function = (
                choice(functions[method], 1)[0]
                if method == "random"
                else functions[method]
            )
            augword.append(function(word=w, degree=degree, position=position)[0])

        # append the augmented sentence along with the identifiers
        all_augword.append([" ".join(augword), sentence] + list(kwargs.values()))

    return all_augword


# run parallel wrapper on __rand_sent_aug__
def rand_sent_aug(
    sentences,
    degree,
    count,
    method="random",
    position="random",
    parallel=True,
    **kwargs
):
    """

    :param sentences:
    :param degree:
    :param count:
    :param method:
    :param position:
    :param parallel:
    :param kwargs:
    :return:
    """
    if parallel:
        function = partial(
            __rand_sent_aug__,
            **kwargs,
            degree=degree,
            method=method,
            count=count,
            position=position,
        )
        return run_parallel(sentences, function)

    else:
        return [
            __rand_sent_aug__(
                sentence, degree, count, method=method, position=position, **kwargs
            )
            for sentence in sentences
        ]


# run_parallel wrapper on __keyboard_sent_aug__
def keyboard_sent_aug(
    sentences,
    degree,
    count,
    method="random",
    position="random",
    parallel=True,
    **kwargs
):
    """

    :param sentences: list of sentences to augment
    :type sentences: list
    :param degree:
    :param count:
    :param method:
    :param position:
    :param parallel:
    :param kwargs:
    :return:
    """
    if parallel:
        function = partial(
            __keyboard_sent_aug__,
            **kwargs,
            degree=degree,
            method=method,
            count=count,
            position=position,
        )
        return run_parallel(sentences, function)

    else:
        return [
            __keyboard_sent_aug__(
                sentence, degree, count, method=method, position=position, **kwargs
            )
            for sentence in sentences
        ]


if __name__ == "__main__":
    data = ["1234 56789 ABCD", "DEEP", "NEURAL CRAVING"]
    rsa = rand_sent_aug(
        sentences=data,
        degree=2,
        count=10,
        method="random",
        position="random",
        parallel=True,
        dummy_identifier_1="generic1",
        dummy_identifier_2="generic2",
    )
    print(rsa)

    data = ["1234 56789 ABCD", "DEEP", "NEURAL CRAVING"]
    ksa = keyboard_sent_aug(
        sentences=data,
        degree=2,
        count=2,
        method="random",
        position="random",
        parallel=True,
        dummy_identifier_1="generic1",
        dummy_identifier_2="generic2",
    )
    print(ksa)
