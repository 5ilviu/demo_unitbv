import numpy as np
from matplotlib import animation
from numpy.random.mtrand import permutation

import alt_rbm
import matplotlib.pyplot as plt

fig = plt.figure()
visual = None
im = plt.imshow(np.zeros((28, 28)), cmap=plt.get_cmap('gray'), animated=True)
i = 0


def get_index():
    global i
    i += 1
    if i >= len(visual):
        i = 0
        print 'restart'
    print(i)
    return i


def updatefig(*args):
    global visual
    mat = np.reshape(visual[get_index()], (28, 28))
    im.set_array(mat)
    return im,


if __name__ == '__main__':
    global visual
    rbm = alt_rbm.RBM(784, 200, learning_rate=0.1)

    # inp = open("matrix{}".format('_test'), "rb")
    # inp = open("matrix{}".format('_better'), "rb")
    # inp = open("matrix{}".format('_even_better'), "rb")
    inp = open("matrix{}".format('_test'), "rb")

    rbm.weights = np.load(inp)
    # good ones are 6 data[6]
    initial = np.random.rand(201).dot(0.7)
    im = plt.imshow(np.random.rand(28, 28), cmap=plt.get_cmap('plasma'), animated=True)
    visual = rbm.daydream(800, initial)
    ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
    plt.show()



