import functools, time
import numpy as np
from numba import prange, jit

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        #        print(f"\nFinished {func.__name__!r} in {run_time:.4f} secs")
        print('\nFinished {!r} in {:.4f} s'.format(func.__name__, run_time))
        return value
    return wrapper_timer


@timer
@jit(parallel = True)
def sampleData(listData, sampleSize, replace = False):
    # Ensure list
    if isinstance(listData, np.ndarray):
        listData = [listData]
    elif isinstance(listData, tuple):
        listData = list(listData)

    # Get indices of the samples
    sampleIdx = np.random.choice(np.arange(len(listData[0])), sampleSize, replace = replace)
    # Go through all provided data
    for i in prange(len(listData)):
        listData[i] = listData[i][sampleIdx]

    print('\nData sampled to {0} with {1} replacement'.format(sampleSize, replace))
    return listData
