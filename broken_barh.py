from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from japanize_matplotlib import japanize
from matplotlib.colors import Colormap

japanize()


class BrokenBarh:
    """不連続な水平バー（broken horizontal bar）を描画するクラス"""

    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 100,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        ylabels: Optional[List[str]] = None,
        colormap: Union[str, Colormap] = "viridis",
    ):
        """
        不連続な水平バーを描画するクラスを初期化する

        Parameters:
        -----------
        figsize : Tuple[int, int], optional
            図のサイズ
        dpi : int, optional
            図の解像度
        title : str, optional
            グラフのタイトル
        xlabel : str, optional
            x軸のラベル
        ylabel : str, optional
            y軸のラベル
        ylabels : List[str], optional
            y軸の各位置に対応するラベル
        colormap : str or Colormap, optional
            カラーマップ
        """
        self.figsize = figsize
        self.dpi = dpi
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.ylabels = ylabels
        self.colormap = colormap

        # フィギュアとアクスの初期化
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # イベントデータの格納用
        self.data = []

    def add_events(
        self,
        xranges: List[Tuple[float, float]],
        yrange: Tuple[float, float],
        label: Optional[str] = None,
        color: Optional[str] = None,
        alpha: float = 0.8,
    ):
        """
        イベントデータを追加する

        Parameters:
        -----------
        xranges : List[Tuple[float, float]]
            各イベントの開始と長さのリスト [(開始位置, 長さ), ...]
        yrange : Tuple[float, float]
            イベントのy位置と高さ (y位置, 高さ)
        label : str, optional
            イベントのラベル（凡例用）
        color : str, optional
            イベントの色（指定しない場合はカラーマップから自動選択）
        alpha : float, optional
            不透明度
        """
        self.data.append({"xranges": xranges, "yrange": yrange, "label": label, "color": color, "alpha": alpha})

    def plot(self):
        """データをプロットする"""
        # 色の自動選択用
        cmap = plt.cm.get_cmap(self.colormap)
        colors = cmap(np.linspace(0, 1, len(self.data)))

        # 各イベントデータをプロット
        for i, event in enumerate(self.data):
            color = event["color"] if event["color"] is not None else colors[i]
            self.ax.broken_barh(
                event["xranges"], event["yrange"], facecolors=color, alpha=event["alpha"], label=event["label"]
            )

        # タイトルとラベルの設定
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)

        # y軸のラベルを設定
        if self.ylabels is not None:
            ypos = [yrange[0] + yrange[1] / 2 for event in self.data for yrange in [event["yrange"]]]
            if len(ypos) == len(self.ylabels):
                self.ax.set_yticks(ypos)
                self.ax.set_yticklabels(self.ylabels)

        # 凡例を表示（ラベルが設定されている場合のみ）
        if any(event["label"] for event in self.data):
            self.ax.legend(loc="upper right")

        # グリッド設定
        self.ax.grid(True, axis="x", linestyle="--", alpha=0.7)

        return self.fig, self.ax

    def save(self, filename: str, **kwargs):
        """図を保存する"""
        self.fig.savefig(filename, **kwargs)

    def show(self):
        """図を表示する"""
        plt.show()
