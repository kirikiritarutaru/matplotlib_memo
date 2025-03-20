"""
積み上げグラフのクラスと関連機能
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from japanize_matplotlib import japanize

japanize()


class StackedGraph:
    """シンプルな積み上げグラフクラス"""

    def __init__(
        self,
        df,
        figsize=(6, 6),
        title="Stacked Graph",
        window_name="plot",
        dpi=100,
    ):
        """初期化

        Parameters
        ----------
        df : pandas.DataFrame
            表示するデータ
        figsize : tuple, optional
            図のサイズ (width, height) in inches
        title : str, optional
            グラフのタイトル
        window_name : str, optional
            OpenCVのウィンドウ名
        dpi : int, optional
            図の解像度（dots per inch）
        """
        if df is None:
            raise ValueError("データフレームを指定してください。")

        # 基本パラメータ
        self.df = df
        self.window_name = window_name
        self.dpi = dpi

        # 図の初期化
        self.fig, self.ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        self.ax.set_title(title)
        self.ax.grid()

        # データの準備
        self.n_rows, self.n_cols = self.df.shape
        self.positions = np.arange(self.n_rows)
        self.colors = plt.get_cmap("tab20c")(np.linspace(0, 1, self.n_cols))

        # グラフの設定
        self.ax.set_yticks(self.positions)
        self.ax.set_yticklabels(self.df.index)
        self.ax.set_xlim([0, self.df.values.sum(axis=1).max() * 1.1])
        self.ax.set_ylim([-1, self.n_rows])

        # 初期描画
        self.fig.canvas.draw()

    def show(self, wait_time=0):
        """グラフを描画して表示する

        Parameters
        ----------
        wait_time : int, optional
            キー入力待ち時間（ミリ秒）、0の場合は無限待ち

        Returns
        -------
        int
            押されたキーのコード
        """
        # グラフを描画
        self.ax.clear()
        self.ax.grid()

        # 設定を再適用
        self.ax.set_yticks(self.positions)
        self.ax.set_yticklabels(self.df.index)
        self.ax.set_xlim([0, self.df.values.sum(axis=1).max() * 1.1])
        self.ax.set_ylim([-1, self.n_rows])

        # 積み上げグラフを描画
        offsets = np.zeros(self.n_rows, dtype=self.df.values.dtype)
        for i in range(len(self.df.columns)):
            bar = self.ax.barh(self.positions, self.df.iloc[:, i], left=offsets, color=self.colors[i])

            # ラベルを追加
            for rect in bar:
                cx = rect.get_x() + rect.get_width() / 2
                cy = rect.get_y() + rect.get_height() / 2
                self.ax.text(
                    cx,
                    cy,
                    self.df.columns[i],
                    color="k",
                    ha="center",
                    va="center",
                )

            offsets += self.df.iloc[:, i]

        # 図を更新
        self.fig.canvas.draw()

        # OpenCVで表示
        np_plot = cv2.cvtColor(np.array(self.fig.canvas.renderer.buffer_rgba()), cv2.COLOR_RGBA2BGR)

        # 表示
        cv2.imshow(self.window_name, np_plot)
        key = cv2.waitKey(wait_time)

        return key

    def add_legend(self, loc="best", **kwargs):
        """凡例を追加

        Parameters
        ----------
        loc : str, optional
            凡例の位置
        **kwargs
            その他のmatplotlib.pyplot.legendに渡すキーワード引数
        """
        self.ax.legend(self.df.columns, loc=loc, **kwargs)

    def save_image(self, filename, format="png"):
        """現在のグラフを画像として保存

        Parameters
        ----------
        filename : str
            保存するファイル名
        format : str, optional
            画像形式（'png', 'jpg'など）

        Returns
        -------
        bool
            保存が成功したかどうか
        """
        try:
            self.fig.savefig(filename, format=format, dpi=self.dpi)
            print(f"画像を {filename} に保存しました")
            return True
        except Exception as e:
            print(f"画像保存エラー: {e}")
            return False

    def cleanup(self):
        """リソースの解放"""
        plt.close(self.fig)
        cv2.destroyAllWindows()

    @property
    def figure(self):
        """Matplotlibの図オブジェクトを取得"""
        return self.fig

    @property
    def axes(self):
        """Matplotlibの軸オブジェクトを取得"""
        return self.ax
