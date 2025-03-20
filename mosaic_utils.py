"""
Matplotlibのmosaicレイアウト関連のユーティリティ
"""

import matplotlib.pyplot as plt
import numpy as np


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


def mosaic_example():
    """mosaicレイアウトの使用例"""

    # 例1: 文字列でレイアウト指定
    axd = plt.figure(constrained_layout=True).subplot_mosaic(
        """
        ABD
        CCD
        """
    )
    identify_axes(axd)
    plt.show()

    # 例2: リストでレイアウト指定、空白セル対応
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

    # 例3: 実際のデータを使用した例
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


def create_grid_layout(rows, cols, figsize=(10, 8), **kwargs):
    """グリッドレイアウトを作成

    Parameters
    ----------
    rows : int
        行数
    cols : int
        列数
    figsize : tuple, optional
        図のサイズ
    **kwargs
        subplot_mosaicに渡す追加引数

    Returns
    -------
    tuple
        (fig, axdict) の組み合わせ
    """
    layout = [[f"{i}_{j}" for j in range(cols)] for i in range(rows)]
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    axdict = fig.subplot_mosaic(layout, **kwargs)
    return fig, axdict


def create_custom_layout(layout_str, figsize=(10, 8), **kwargs):
    """カスタムレイアウトを作成

    Parameters
    ----------
    layout_str : str
        レイアウト文字列（例: "AB\nCC"）
    figsize : tuple, optional
        図のサイズ
    **kwargs
        subplot_mosaicに渡す追加引数

    Returns
    -------
    tuple
        (fig, axdict) の組み合わせ
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    axdict = fig.subplot_mosaic(layout_str, **kwargs)
    return fig, axdict
