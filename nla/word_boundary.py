from random import sample
from numpy.random import binomial, randint
from numpy import unique, concatenate
from nla.parallelize import *
from functools import partial


def split(number, nparts):
    """
    split a number in n parts
    :param number: the number to split
    :type number: int
    :param nparts: number of parts (n)
    :type nparts: int
    :return:
    """
    return (
        [number // nparts for i in range(nparts)]
        if number % nparts == 0
        else [
            number // nparts + 1
            if i >= nparts - (number % nparts)
            else number // nparts
            for i in range(nparts)
        ]
    )


def __word_boundary__(query, count, degree=1, **kwargs):
    """
    :param query: sentence to augment
    :type query: str
    :param count: number of output for each query
    :type count: int
    :param degree: degree of augmentation, takes value between 0 and 1
    :type degree: float
    :return:
    """
    assert 0 <= degree <= 1, "degree argument takes values between 0 and 1"

    result = set()

    query, actual = "".join(sample(query.split(), len(query.split()))), query

    # query length
    qlen = len(query)
    result.add((query, actual) + tuple(kwargs.values())) if len(
        actual.split()
    ) > 1 else 0

    # number of spaces to add
    numspace = (
        randint(3, qlen - 1)
        if qlen >= 10
        else 3
        if qlen in range(6, 10)
        else 2
        if qlen > 1
        else 0
    )

    # generating random indices for putting in spaces, dimension (count, numspace)
    idx = (
        concatenate(
            [
                binomial(
                    1,
                    (0.30 + (0.20 * i)) * degree,
                    (count, numspace),
                )
                for i, count in enumerate(split(count, 4))
            ]
        )
        * randint(1, qlen, (count, numspace), dtype="int64")
    )

    idx = [unique(ids[ids != 0]) - 1 for ids in idx]

    # insert spaces at ids in idx
    [
        result.add(
            (
                "".join(
                    [
                        char if i not in ids else char + " "
                        for i, char in enumerate(query)
                    ]
                ).strip(),
                actual,
            )
            + tuple(kwargs.values())
        )
        for ids in idx
    ]

    return list(result)


def word_boundary(queries, count, degree, parallel=True, **kwargs):
    """
    run augmentation on list of sentences
    :param queries: sentences to augment
    :type queries: list
    :param count: number of output for each query
    :type count: int
    :param degree: degree of augmentation, takes value between 0 and 1
    :type degree: float
    :param parallel: run in parallel
    :type parallel: bool
    :param kwargs:
    :return:
    """
    if parallel:
        function = partial(
            __word_boundary__,
            **kwargs,
            degree=degree,
            count=count,
        )
        return run_parallel(queries, function)

    else:
#         return [
#             __word_boundary__(word, degree=degree, count=count, **kwargs)
#             for word in queries
#         ]
        output = []
        
        for word in queries:
            output.extend(__word_boundary__(word, degree=degree, count=count, **kwargs))
            
        return output


if __name__ == "__main__":
    # serial execution
    print(
        __word_boundary__(
            query="DEEP NEURAL CRAVING", count=10, degree=0.5, args="generic"
        )
    )

    # parallel execution
    data = ["DEEP NEURAL CRAVING", "DEEP CRAVING", "NEURAL CRAVING"]
    print(
        word_boundary(
            data,
            count=2,
            degree=0.6,
            parallel=True,
            dummy_identifier_1="generic1",
            dummy_identifier_2="generic2",
        )
    )
