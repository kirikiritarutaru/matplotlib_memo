from typing import Dict, List, Optional, Tuple, Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from japanize_matplotlib import japanize
from matplotlib.colors import Colormap

japanize()


class BarRace:
    """バーレースアニメーションを作成するクラス"""

    def __init__(
        self,
        data: Dict[str, List[float]],
        labels: List[str],
        title: str = "",
        interval: int = 100,
        top_n: Optional[int] = None,
        colormap: Union[str, Colormap] = "viridis",
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 100,
        xlabel: str = "",
        ylabel: str = "",
    ):
        """
        バーレースアニメーションを初期化する

        Parameters:
        -----------
        data : Dict[str, List[float]]
            各時点でのデータ値（キー：カテゴリ名、値：各時点での値のリスト）
        labels : List[str]
            各フレームのラベル（時間や日付など）
        title : str, optional
            グラフのタイトル
        interval : int, optional
            アニメーションのフレーム間隔（ミリ秒）
        top_n : int, optional
            表示する上位n件のデータ（Noneの場合は全て表示）
        colormap : str or Colormap, optional
            カラーマップ
        figsize : Tuple[int, int], optional
            図のサイズ
        dpi : int, optional
            図の解像度
        xlabel : str, optional
            x軸のラベル
        ylabel : str, optional
            y軸のラベル
        """
        self.data = data
        self.labels = labels
        self.title = title
        self.interval = interval
        self.top_n = top_n
        self.colormap = colormap
        self.figsize = figsize
        self.dpi = dpi
        self.xlabel = xlabel
        self.ylabel = ylabel

        # 検証
        self._validate_data()

        # フィギュアとアクスの初期化
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        self.ani = None

    def _validate_data(self):
        """データの整合性を検証する"""
        n_frames = len(self.labels)

        # 全てのカテゴリのデータ長が同じかチェック
        for category, values in self.data.items():
            if len(values) != n_frames:
                raise ValueError(
                    f"カテゴリ '{category}' のデータ長 ({len(values)}) がラベル数 ({n_frames}) と一致しません"
                )

    def _get_frame_data(self, frame_idx: int) -> Tuple[List[str], List[float]]:
        """指定フレームのデータを取得し、必要に応じてソートとフィルタリングを行う"""
        categories = list(self.data.keys())
        values = [self.data[cat][frame_idx] for cat in categories]

        # 値に基づいてソート
        sorted_data = sorted(zip(categories, values), key=lambda x: x[1], reverse=True)

        # 上位N件に制限
        if self.top_n is not None:
            sorted_data = sorted_data[: self.top_n]

        # 結果を分解
        sorted_categories, sorted_values = zip(*sorted_data) if sorted_data else ([], [])

        return list(sorted_categories), list(sorted_values)

    def _update_frame(self, frame_idx: int):
        """各フレームの更新処理"""
        self.ax.clear()

        categories, values = self._get_frame_data(frame_idx)

        # バーの描画
        bars = self.ax.barh(
            categories, values, color=plt.cm.get_cmap(self.colormap)(np.linspace(0, 1, len(categories)))
        )

        # 値のアノテーション
        for bar, value in zip(bars, values):
            self.ax.text(value + max(values) * 0.02, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center")

        # タイトルとラベルの設定
        frame_title = f"{self.title} - {self.labels[frame_idx]}" if self.title else self.labels[frame_idx]
        self.ax.set_title(frame_title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)

        # グリッド設定
        self.ax.grid(axis="x", linestyle="--", alpha=0.7)

        # 最大値に基づいてx軸の範囲を設定
        self.ax.set_xlim(0, max(values) * 1.1)

        return bars

    def animate(self) -> animation.FuncAnimation:
        """アニメーションを作成して返す"""
        self.ani = animation.FuncAnimation(
            self.fig, self._update_frame, frames=len(self.labels), interval=self.interval, blit=False
        )
        return self.ani

    def save(self, filename: str, fps: int = 10, **kwargs):
        """アニメーションを保存する"""
        if self.ani is None:
            self.animate()

        writer = animation.FFMpegWriter(fps=fps, **kwargs)
        self.ani.save(filename, writer=writer)

    def show(self):
        """アニメーションを表示する"""
        if self.ani is None:
            self.animate()
        plt.show()
