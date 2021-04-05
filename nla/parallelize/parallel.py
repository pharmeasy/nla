from multiprocessing.pool import Pool
import multiprocessing


# function to parallelize takes a single argument
def run_parallel(data, function):
    """
    parallelize
    :param data: iterator
    :param function:
    :return:
    """
    output = []
    num_cores = min(multiprocessing.cpu_count(), len(data))
    data = (row for row in data)

    with Pool(processes=num_cores) as pool:
        for result in pool.map(function, data):
            output.extend(result)

    return output
