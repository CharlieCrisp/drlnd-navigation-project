import time

import numpy as np
import matplotlib.pyplot as plt

plt.ion()


def draw(fig):
    fig.canvas.draw()
    fig.canvas.flush_events()


if __name__ == "__main__":
    xs = np.arange(0, 10)
    ys = xs ** 2

    fig, ax = plt.subplots()
    line, = ax.plot(xs, ys)
    draw(fig)

    for i in range(10):
        xs = np.append(xs, 10 + i)
        ys = np.append(ys, (10 + i) ** 2)

        ax.set_xlim(min(xs), max(xs))
        ax.set_ylim(min(ys), max(ys))

        line.set_xdata(xs)
        line.set_ydata(ys)

        draw(fig)

        time.sleep(1)

