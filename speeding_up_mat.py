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
        floor_plot = cv2.cvtColor(
            np.array(fig.canvas.renderer.buffer_rgba()), cv2.COLOR_RGBA2BGR
        )


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
                cx, cy, df.columns[i], color="k", ha="center", va="center", animated=True
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
        np_plot = cv2.cvtColor(np.array(fig.canvas.renderer.buffer_rgba()), cv2.COLOR_RGBA2BGR)

        cv2.imshow('plot', np_plot)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


if __name__ == '__main__':
    # vis_stacked_graph()
    stacked_graph_race()
