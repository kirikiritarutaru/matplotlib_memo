import cv2
import matplotlib.pyplot as plt
import numpy as np


def init_mpl(x_range, y_range, figsize=(6, 6)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect('equal')
    ax.grid()
    fig.canvas.draw()
    plt_bg = fig.canvas.copy_from_bbox(fig.bbox)
    return fig, ax, plt_bg


if __name__ == '__main__':
    fig, ax, plt_bg = init_mpl(x_range=[0, 6], y_range=[-1, 1])
    x = np.linspace(0, 2*np.pi, 100)
    (ln, ) = ax.plot(x, np.sin(x), animated=True)
    plt.show(block=False)
    plt.pause(0.1)
    j = 0
    while True:
        fig.canvas.restore_region(plt_bg)

        ln.set_ydata(np.sin(x + (j / 100) * np.pi))
        ax.draw_artist(ln)
        j += 1

        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()
        # opencvで表示用にnumpyの形にして書き出し
        floor_plot = cv2.cvtColor(
            np.array(fig.canvas.renderer.buffer_rgba()), cv2.COLOR_RGBA2BGR
        )
