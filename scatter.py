"""
シンプルな高速散布図表示用クラス

このモジュールは、Matplotlibを使った高速な散布図アニメーション表示を
シンプルなインターフェースで実現するためのクラスを提供します。
"""

import queue
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from japanize_matplotlib import japanize

japanize()


class Scatter:
    """シンプルな高速散布図表示用クラス

    データの管理はクラス外部で行い、プロットするデータの塊を
    渡すことでアニメーションを表示します。

    Examples
    --------
    >>> plt = Scatter((0, 10), (-1, 1), title="ランダム散布図")
    >>> plt.set_xlabel("X軸")
    >>> plt.set_ylabel("Y軸")
    >>> for i in range 100):
    >>>     x = np.random.rand(100) * 10
    >>>     y = np.random.rand(100) * 2 - 1
    >>>     colors = np.random.rand(100)
    >>>     sizes = np.random.rand(100) * 100 + 10
    >>>     if plt.plot(x, y, colors=colors, sizes=sizes) == 27:  # ESCキーで終了
    >>>         break
    >>> plt.save_animation("scatter_animation.mp4")
    >>> plt.close()  # リソースの解放
    """

    def __init__(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        figsize: Tuple[int, int] = (8, 6),
        title: Optional[str] = None,
        window_name: str = "scatter",
        dpi: int = 100,
        color_map: str = "viridis",
        marker: str = "o",
        alpha: float = 0.7,
        edge_colors: str = None,
        max_frames: int = 1000,
        auto_clear: bool = True,
        downsample: bool = False,
        max_points: int = 1000,
        use_threading: bool = False,
        optimize_memory: bool = True,
    ):
        """初期化処理

        Parameters
        ----------
        x_range : tuple
            X軸の表示範囲 (min, max)
        y_range : tuple
            Y軸の表示範囲 (min, max)
        figsize : tuple, optional
            図のサイズ (width, height) in inches
        title : str, optional
            グラフのタイトル
        window_name : str, optional
            OpenCVのウィンドウ名
        dpi : int, optional
            図の解像度（dots per inch）
        color_map : str, optional
            デフォルトのカラーマップ
        marker : str, optional
            デフォルトのマーカー形状 (例: 'o', 's', '^', '*')
        alpha : float, optional
            透明度 (0.0〜1.0)
        edge_colors : str, optional
            マーカーの縁の色
        max_frames : int, optional
            保存する最大フレーム数（これを超えると古いフレームは破棄または自動クリア）
        auto_clear : bool, optional
            最大フレーム数に達したときに自動的にフレームをクリアするかどうか
        downsample : bool, optional
            大量のデータポイントがある場合に自動的にダウンサンプリングするかどうか
        max_points : int, optional
            ダウンサンプリング時の最大点数
        use_threading : bool, optional
            動画保存や画面描画をバックグラウンドスレッドで行うかどうか
        optimize_memory : bool, optional
            メモリ使用量を最適化するためにフレームのコピー方法を変更するかどうか
        """
        # Matplotlib図の作成
        self.fig, self.ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        # 設定値を保存
        self._x_range = x_range
        self._y_range = y_range
        self._title = title
        self._window_name = window_name
        self._color_map = color_map
        self._alpha = alpha
        self._marker = marker
        self._edge_colors = edge_colors

        # 軸の設定
        self.ax.set_xlim(x_range)
        self.ax.set_ylim(y_range)
        self.ax.grid(True)

        if title:
            self.ax.set_title(title)

        # 単一グループモード - 空の散布図を初期化
        self.scatter = self.ax.scatter(
            [],
            [],
            c=[],
            cmap=color_map,
            marker=marker,
            alpha=alpha,
            edgecolors=edge_colors,
            animated=True,
        )

        # OpenCV用の設定
        self.window_name = window_name

        # 録画用の設定
        self.frames = []
        self.max_frames = max_frames
        self.auto_clear = auto_clear

        # 初期描画して背景を保存
        self.fig.canvas.draw()
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

        # ラベル追跡用
        self._xlabel = ""
        self._ylabel = ""

        # 高速化用の設定
        self.downsample = downsample
        self.max_points = max_points
        self.use_threading = use_threading
        self.optimize_memory = optimize_memory

        # スレッディング用のキュー
        if use_threading:
            self.frame_queue = queue.Queue(maxsize=30)
            self.recording = True
            self.display_thread = threading.Thread(target=self._display_thread, daemon=True)
            self.display_thread.start()

        # フレーム使用メモリの最適化
        if optimize_memory:
            self.frame_shape = None

    @property
    def title(self) -> Optional[str]:
        """グラフのタイトルを取得"""
        return self._title

    @title.setter
    def title(self, value: str):
        """グラフのタイトルを設定"""
        self._title = value
        self.ax.set_title(value)
        self._update_background()

    @property
    def x_range(self) -> Tuple[float, float]:
        """X軸の範囲を取得"""
        return self._x_range

    @x_range.setter
    def x_range(self, value: Tuple[float, float]):
        """X軸の範囲を設定"""
        self._x_range = value
        self.ax.set_xlim(value)
        self._update_background()

    @property
    def y_range(self) -> Tuple[float, float]:
        """Y軸の範囲を取得"""
        return self._y_range

    @y_range.setter
    def y_range(self, value: Tuple[float, float]):
        """Y軸の範囲を設定"""
        self._y_range = value
        self.ax.set_ylim(value)
        self._update_background()

    def _update_background(self):
        """背景を更新"""
        self.fig.canvas.draw()
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def _downsample_data(self, x_data, y_data, colors=None, sizes=None):
        """データを間引く

        大量のデータポイントがある場合に間引いて表示速度を向上させる

        Parameters
        ----------
        x_data : array-like
            X座標データの配列
        y_data : array-like
            Y座標データの配列
        colors : array-like, optional
            色データの配列
        sizes : array-like, optional
            サイズデータの配列

        Returns
        -------
        tuple
            (間引かれたX座標配列, 間引かれたY座標配列, 間引かれた色配列, 間引かれたサイズ配列)
        """
        data_length = len(x_data)

        if not self.downsample or data_length <= self.max_points:
            return x_data, y_data, colors, sizes

        # データを間引く
        indices = np.linspace(0, data_length - 1, self.max_points, dtype=int)

        downsampled_x = x_data[indices]
        downsampled_y = y_data[indices]
        downsampled_colors = colors[indices] if colors is not None else None
        downsampled_sizes = sizes[indices] if sizes is not None else None

        return downsampled_x, downsampled_y, downsampled_colors, downsampled_sizes

    def _display_thread(self):
        """バックグラウンドで画面更新を行うスレッド"""
        while self.recording:
            try:
                # キューからフレームを取得
                frame, window_name, wait_time = self.frame_queue.get(timeout=0.1)
                if frame is not None:
                    cv2.imshow(window_name, frame)
                    cv2.waitKey(wait_time)
                self.frame_queue.task_done()
            except queue.Empty:
                # キューが空の場合は少し待機
                time.sleep(0.01)
            except Exception as e:
                print(f"表示スレッドエラー: {e}")

    def plot(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        wait_time: int = 1,
        colors: Optional[np.ndarray] = None,
        sizes: Optional[np.ndarray] = None,
    ) -> int:
        """散布図をプロットして表示

        Parameters
        ----------
        x_data : np.ndarray
            X座標データの配列
        y_data : np.ndarray
            Y座標データの配列
        wait_time : int, optional
            表示の待機時間（ミリ秒）
        colors : np.ndarray, optional
            各点の色
        sizes : np.ndarray, optional
            各点のサイズ

        Returns
        -------
        int
            押されたキーのコード（なければ-1、ESCキーは27）
        """
        # データの間引き処理
        if self.downsample:
            x_data, y_data, colors, sizes = self._downsample_data(x_data, y_data, colors, sizes)

        # 背景を復元
        self.fig.canvas.restore_region(self.bg)

        # データを設定
        self.scatter.set_offsets(np.c_[x_data, y_data])
        if colors is not None:
            self.scatter.set_array(colors)
        if sizes is not None:
            self.scatter.set_sizes(sizes)
        self.ax.draw_artist(self.scatter)

        # 画面更新
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

        # OpenCVウィンドウに表示
        np_plot = cv2.cvtColor(np.array(self.fig.canvas.renderer.buffer_rgba()), cv2.COLOR_RGBA2BGR)

        # メモリ最適化: フレームの形状を保存し、必要なら再利用
        if self.optimize_memory:
            if self.frame_shape is None:
                self.frame_shape = np_plot.shape

            # 最適化版のフレーム記録
            if len(self.frames) < self.max_frames:
                self.frames.append(np_plot.copy())
            else:
                if self.auto_clear:
                    self.clear_frames()
                    self.frames.append(np_plot.copy())
                    print(f"警告: フレーム数が{self.max_frames}を超えたため、自動的にクリアしました。")
                else:
                    # 古いフレームを上書き再利用
                    self.frames.pop(0)
                    self.frames.append(np_plot.copy())
        else:
            # 従来のフレーム記録方法
            self.frames.append(np_plot.copy())

            # 最大フレーム数をチェック
            if len(self.frames) > self.max_frames:
                if self.auto_clear:
                    self.clear_frames()
                    print(f"警告: フレーム数が{self.max_frames}を超えたため、自動的にクリアしました。")
                else:
                    # 古いフレームを削除
                    self.frames.pop(0)

        # スレッド使用か直接表示か
        key = -1
        if self.use_threading:
            try:
                # キューに追加して非同期処理
                self.frame_queue.put((np_plot, self.window_name, wait_time), block=False)
            except queue.Full:
                # キューがいっぱいの場合は直接表示
                cv2.imshow(self.window_name, np_plot)
                key = cv2.waitKey(wait_time)
        else:
            # 通常の表示
            cv2.imshow(self.window_name, np_plot)
            key = cv2.waitKey(wait_time)

        return key

    def add_legend(self, label: str, **kwargs):
        """凡例を追加

        Parameters
        ----------
        label : str
            凡例のラベル
        **kwargs
            matplotlibのlegendに渡すその他のパラメータ
        """
        self.ax.legend([self.scatter], [label], **kwargs)
        self._update_background()

    def add_colorbar(self, label: Optional[str] = None, **kwargs) -> None:
        """カラーバーを追加

        Parameters
        ----------
        label : str, optional
            カラーバーのラベル
        **kwargs
            matplotlibのcolorbarに渡すその他のパラメータ
        """
        cbar = self.fig.colorbar(self.scatter, ax=self.ax, **kwargs)
        if label:
            cbar.set_label(label)
        self._update_background()

    def set_xlabel(self, label: str):
        """X軸のラベルを設定

        Parameters
        ----------
        label : str
            X軸のラベル
        """
        self._xlabel = label
        self.ax.set_xlabel(label)
        self._update_background()

    def set_ylabel(self, label: str):
        """Y軸のラベルを設定

        Parameters
        ----------
        label : str
            Y軸のラベル
        """
        self._ylabel = label
        self.ax.set_ylabel(label)
        self._update_background()

    def save_animation(self, filename: str, fps: int = 30, parallel: bool = True) -> bool:
        """アニメーションを動画として保存

        Parameters
        ----------
        filename : str
            保存するファイル名
        fps : int, optional
            フレームレート（1秒あたりのフレーム数）
        parallel : bool, optional
            バックグラウンドで動画を保存するかどうか (use_threading=True の場合のみ有効)

        Returns
        -------
        bool
            保存が成功したかどうか
        """
        if not self.frames:
            print("保存するフレームがありません")
            return False

        # 並列処理で保存
        if parallel and self.use_threading:
            save_thread = threading.Thread(target=self._save_animation_thread, args=(filename, fps), daemon=True)
            save_thread.start()
            print(f"バックグラウンドでアニメーションの保存を開始しました: {filename}")
            return True
        else:
            # 通常の保存処理
            try:
                # 出力パス
                output_path = Path(filename)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # MP4形式で保存
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                height, width, _ = self.frames[0].shape
                video_path = output_path.with_suffix(".mp4")

                video = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

                for frame in self.frames:
                    video.write(frame)

                video.release()
                print(f"アニメーションを{video_path}に保存しました (フレーム数: {len(self.frames)})")
                return True

            except Exception as e:
                print(f"保存エラー: {e}")
                return False

    def _save_animation_thread(self, filename: str, fps: int):
        """バックグラウンドで動画を保存するスレッド"""
        try:
            # 出力パス
            output_path = Path(filename)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # MP4形式で保存
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            height, width, _ = self.frames[0].shape
            video_path = output_path.with_suffix(".mp4")

            video = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

            for frame in self.frames:
                video.write(frame)

            video.release()
            print(f"バックグラウンド処理: アニメーションを{video_path}に保存しました (フレーム数: {len(self.frames)})")
        except Exception as e:
            print(f"バックグラウンド保存エラー: {e}")

    def close(self):
        """リソースを解放"""
        if self.use_threading:
            # スレッドを終了
            self.recording = False
            if hasattr(self, "display_thread") and self.display_thread.is_alive():
                self.display_thread.join(timeout=1.0)  # 最大1秒待機

        plt.close(self.fig)
        cv2.destroyAllWindows()

    def clear_frames(self):
        """記録されたフレームをすべて削除"""
        self.frames = []
        print("フレームデータをクリアしました。")

    def get_frame_count(self) -> int:
        """現在記録されているフレーム数を取得

        Returns
        -------
        int
            記録されているフレーム数
        """
        return len(self.frames)

    def get_plot_image(self) -> np.ndarray:
        """現在のプロット画像をNumPy配列として取得

        散布図をOpenCV/NumPy互換の画像配列として返します。
        この画像はBGR形式のNumPy配列（shape: height x width x 3）です。

        Returns
        -------
        np.ndarray
            プロット画像のNumPy配列（BGR形式）

        Examples
        --------
        >>> plt = Scatter((0, 10), (-1, 1))
        >>> x = np.random.rand(100) * 10
        >>> y = np.random.rand(100) * 2 - 1
        >>> plt.plot(x, y)
        >>> img = plt.get_plot_image()
        >>> cv2.imwrite("scatter_plot.png", img)  # 画像をファイルに保存
        """
        # 背景を復元して最新の状態を描画
        self.fig.canvas.draw()

        # Matplotlibのキャンバスから画像データを取得しOpenCV形式に変換
        img_array = np.array(self.fig.canvas.renderer.buffer_rgba())
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

        return img_bgr
