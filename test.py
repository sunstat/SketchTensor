import numpy as np
import time


def test_generator():
    np.random.seed(1)
    for i in range(10):
        _ = np.random.normal(0,1,1)

    for j in range(10):
        yield np.random.normal(0,1,(10000,5000))

def test_return():
    np.random.seed(1)
    for i in range(10):
        _ = np.random.normal(0,1,1)

    return np.random.normal(0,1,(10000,5000,10))

def test_generator():
    start_time = time.time()
    rm_generator = test_generator()
    for num in rm_generator:
        _ = 1+1
    print("--- %s seconds ---" % (time.time() - start_time))

    print("finish")
    start_time = time.time()
    test_return()
    print("--- %s seconds ---" % (time.time() - start_time))


def test_fun_arguments(arg1, arg2):
    print(arg1)
    print(arg2)



if __name__ == "__main__":
    #test_generator()
    test_fun_arguments(arg2=1, arg1=2)

