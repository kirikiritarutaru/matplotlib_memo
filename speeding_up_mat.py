import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def init_mpl(x_range, y_range, figsize=(6, 6)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect('equal')
    ax.grid()
    fig.canvas.draw()
    plt_bg = fig.canvas.copy_from_bbox(fig.bbox)
    return fig, ax, plt_bg


def fast_mpl():
    fig, ax, plt_bg = init_mpl(x_range=[0, 6], y_range=[-1, 1])
    x = np.linspace(0, 2 * np.pi, 100)
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
        np_plot = cv2.cvtColor(
            np.array(fig.canvas.renderer.buffer_rgba()), cv2.COLOR_RGBA2BGR
        )
        cv2.imshow('plot', np_plot)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


def get_data():
    data = np.array(
        [[156, 128, 285], [286, 220, 297], [218, 235, 137], [160, 298, 112]]
    )
    row_labels = ["2017", "2018", "2019", "2020"]
    col_labels = ["Android", "iOS", "Windows"]

    return pd.DataFrame(data, index=row_labels, columns=col_labels)


def vis_stacked_graph():
    df = get_data()

    n_rows, n_cols = df.shape
    positions = np.arange(n_rows)
    offsets = np.zeros(n_rows, dtype=df.values.dtype)
    colors = plt.get_cmap("tab20c")(np.linspace(0, 1, n_cols))

    fig, ax = plt.subplots()
    ax.set_yticks(positions)
    ax.set_yticklabels(df.index)
    ax.set_xlim([0, 800])
    ax.set_ylim([-1, 4])

    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(fig.bbox)

    fig.canvas.restore_region(bg)

    for i in range(len(df.columns)):
        # 棒グラフを描画する。
        bar = ax.barh(positions, df.iloc[:, i], left=offsets, color=colors[i])
        offsets += df.iloc[:, i]

        # 棒グラフのラベルを描画する。
        for rect in bar:
            cx = rect.get_x() + rect.get_width() / 2
            cy = rect.get_y() + rect.get_height() / 2
            ax.draw_artist(ax.add_patch(rect))
            text_ax = ax.text(
                cx, cy, df.columns[i], color="k",
                ha="center", va="center", animated=True
            )
            ax.draw_artist(text_ax)

    fig.canvas.blit(fig.bbox)
    fig.canvas.flush_events()
    np_plot = cv2.cvtColor(
        np.array(fig.canvas.renderer.buffer_rgba()),
        cv2.COLOR_RGBA2BGR
    )

    cv2.imshow('plot', np_plot)
    cv2.waitKey(0)


def stacked_graph_race():
    data = np.array(
        [[10, 290, 280], [10, 100, 90], [200, 300, 100], [150, 250, 100]]
    )
    df = pd.DataFrame(data, columns=['start', 'end', 'diff'])

    n_rows, n_cols = df.shape
    positions = np.arange(n_rows)
    colors = plt.get_cmap("tab20c")(np.linspace(0, 1, n_cols))

    fig, ax = plt.subplots()
    ax.set_yticks(positions)
    ax.set_yticklabels(df.index)
    ax.set_xlim([0, int(df['end'].max())])
    ax.set_ylim([-1, n_cols + 1])
    ax.invert_yaxis()
    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(fig.bbox)

    for i in range(df['end'].max()):
        fig.canvas.restore_region(bg)

        df_s = (i - df['start']).clip(0, df['diff'])
        bar = ax.barh(positions, df_s, left=df['start'], color=colors)

        for rect in bar:
            ax.draw_artist(rect)
        ax.draw_artist(ax.axvline(i, color='red'))

        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()
        np_plot = cv2.cvtColor(
            np.array(fig.canvas.renderer.buffer_rgba()),
            cv2.COLOR_RGBA2BGR
        )

        cv2.imshow('plot', np_plot)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


def broken_barh_example():
    fig, ax = plt.subplots()
    ax.broken_barh([(110, 30), (150, 10)], (10, 9), facecolors='tab:blue')
    ax.broken_barh([(10, 50), (100, 20), (130, 10)], (20, 9),
                   facecolors=('tab:orange', 'tab:green', 'tab:red'))
    ax.set_ylim(5, 35)
    ax.set_xlim(0, 200)
    ax.set_xlabel('seconds since start')
    ax.set_yticks([15, 25], labels=['Bill', 'Jim'])
    ax.grid(True)
    plt.show()


def broken_barh_race(
        data=np.array([[10,  280], [10,  90], [200, 100], [150, 100]])
):
    df = pd.DataFrame(data, columns=['start', 'diff'])
    n_rows, n_cols = df.shape
    positions = np.arange(n_rows)
    colors = plt.get_cmap("tab20c")(np.linspace(0, 1, n_rows))
    fig, ax = plt.subplots()

    x_limit = 300
    ax.set_xlabel('time')
    ax.get_yaxis().set_visible(False)
    ax.set_ylim(0, n_rows)
    ax.set_xlim(0, x_limit)
    ax.invert_yaxis()
    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(fig.bbox)

    for i in range(x_limit):
        fig.canvas.restore_region(bg)
        # barhで書くのとやってること変わらんな…
        for j in range(n_rows):
            if i < df['start'][j]:
                continue
            xrange = [(df['start'][j], min(i-df['start'][j], df['diff'][j]))]
            bar = ax.broken_barh(
                xrange, (positions[j], 0.5), facecolors=colors[j]
            )
            ax.draw_artist(bar)

        ax.draw_artist(ax.axvline(i, color='red'))
        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()
        np_plot = cv2.cvtColor(
            np.array(fig.canvas.renderer.buffer_rgba()),
            cv2.COLOR_RGBA2BGR
        )

        cv2.imshow('plot', np_plot)
        key = cv2.waitKey(50)
        if key == ord('q'):
            break


# Helper function used for visualization in the following examples
def identify_axes(ax_dict, fontsize=48):
    """
    Helper to identify the Axes in the examples below.

    Draws the label in a large font in the center of the Axes.

    Parameters
    ----------
    ax_dict : dict[str, Axes]
        Mapping between the title / label and the Axes.
    fontsize : int, optional
        How big the label should be.
    """
    kw = dict(ha="center", va="center", fontsize=fontsize, color="darkgrey")
    for k, ax in ax_dict.items():
        ax.text(0.5, 0.5, k, transform=ax.transAxes, **kw)


# 参考:https://matplotlib.org/stable/tutorials/provisional/mosaic.html
def mosaic_example():
    axd = plt.figure(constrained_layout=True).subplot_mosaic(
        """
        ABD
        CCD
        """
    )
    identify_axes(axd)
    plt.show()

    axd = plt.figure(constrained_layout=True).subplot_mosaic(
        [
            ["main", "zoom"],
            ["main", "BLANK"],
        ],
        empty_sentinel="BLANK",
        gridspec_kw={"width_ratios": [2, 1]},
    )
    identify_axes(axd)
    plt.show()

    hist_data = np.random.randn(1_500)
    fig = plt.figure(constrained_layout=True)
    ax_dict = fig.subplot_mosaic(
        [
            ["bar", "plot"],
            ["hist", "image"],
        ],
    )
    ax_dict["bar"].bar(["a", "b", "c"], [5, 7, 9])
    ax_dict["plot"].plot([1, 2, 3])
    ax_dict["hist"].hist(hist_data)
    ax_dict["image"].imshow([[1, 2], [2, 1]])
    identify_axes(ax_dict)
    plt.show()


if __name__ == '__main__':
    # vis_stacked_graph()
    # stacked_graph_race()
    # broken_barh_example()
    # broken_barh_race()
    mosaic_example()
