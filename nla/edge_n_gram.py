from itertools import product
from random import sample
from nla.parallelize import *
from functools import partial


def __edge_n_gram__(query, count, degree=4, **kwargs):
    """
    computes edge n-gram for each item in the list
    :param query: query to augment
    :type query: str
    :param count: number of output for each query
    :type count: int
    :param degree: number of characters to remove from each word
    :type degree: int
    :return:
    """
    # minimum length of a word after augmentation
    threshold = 3

    # remove ~'degree' number of characters from every word in the sentence
    y = [
        {
            w[: (-1 * i)] if (len(w) > i) & (len(w[: (-1 * i)]) >= threshold) else w
            for i in range(1, degree + 1)
        }
        for w in query.split()
    ]

    # add the augmented sentence along with the identifiers
    new = [(" ".join(a), query) + tuple(kwargs.values()) for a in product(*y)]
    new = sample(new, count) if count < len(new) else new
    
    return new


def edge_n_gram(queries, count, degree, parallel=True, **kwargs):
    """
    run the augmentation on list of sentences
    :param queries: sentences to augment
    :type queries: list
    :param count: number of output for each query
    :type count: int
    :param degree: number of characters to remove from each word
    :type degree: int
    :param parallel: run in parallel
    :type parallel: bool
    :param kwargs:
    :return:
    """
    if parallel:
        function = partial(
            __edge_n_gram__,
            **kwargs,
            degree=degree,
            count=count,
        )
        return run_parallel(queries, function)

    else:
        return [
            __edge_n_gram__(word, degree=degree, count=count, **kwargs)
            for word in queries
        ]


if __name__ == "__main__":
    # serial execution
    print(__edge_n_gram__(query="DEEP NEURAL CRAVING", count=10, degree=4, args="generic"))

    # parallel execution
    data = ["DEEP NEURAL CRAVING", "DEEP CRAVING", "NEURAL CRAVING"]
    print(
        edge_n_gram(
            data,
            count=3,
            degree=2,
            parallel=True,
            dummy_identifier_1="generic1",
            dummy_identifier_2="generic2",
        )
    )
