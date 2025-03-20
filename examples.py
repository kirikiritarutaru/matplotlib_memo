import time
from pathlib import Path

import numpy as np
import pandas as pd

from bar_race import BarRace
from broken_barh import BrokenBarh
from plotter import Plotter
from stacked_graph import StackedGraph


def ensure_output_dir():
    """出力ディレクトリが存在することを確認"""
    output_dir = Path(".") / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def example_sine_wave():
    """基本的な正弦波アニメーションの例"""
    print("正弦波アニメーションの例")

    # プロッターを初期化
    plotter = Plotter(
        x_range=(0, 10),
        y_range=(-1.5, 1.5),
        title="サイン波アニメーション",
        figsize=(10, 6),
    )

    # 軸ラベルを設定
    plotter.set_xlabel("時間 (秒)")
    plotter.set_ylabel("振幅")

    # 100フレームのアニメーションを作成
    for i in range(100):
        x = np.linspace(0, 10, 1000)
        y = np.sin(x - i / 10)  # 移動する正弦波

        # プロットして、キー入力をチェック
        key = plotter.plot(x, y)
        if key == 27:  # ESCキーで終了
            break

    # 出力ディレクトリを確保
    output_dir = ensure_output_dir()

    # アニメーションを保存
    output_path = output_dir / "sine_wave.mp4"
    plotter.save_animation(str(output_path))

    # リソースを解放
    plotter.close()


def example_lissajous():
    """リサージュ図形のアニメーション例"""
    print("リサージュ図形のアニメーション例")

    # プロッターを初期化 (-1.5から1.5の範囲)
    plotter = Plotter(
        x_range=(-1.5, 1.5),
        y_range=(-1.5, 1.5),
        title="リサージュ図形",
        line_style="-",
        color="r",
        figsize=(8, 8),
    )

    # 軸ラベルを設定
    plotter.set_xlabel("X軸")
    plotter.set_ylabel("Y軸")

    # パラメータを変化させながらリサージュ図形を描画
    for delta in range(100):
        t = np.linspace(0, 2 * np.pi, 1000)
        a, b = 3, 4  # 周波数比
        delta_rad = delta * np.pi / 50  # 位相差を徐々に変化

        x = np.sin(a * t + delta_rad)
        y = np.sin(b * t)

        key = plotter.plot(x, y)
        if key == 27:  # ESCキーで終了
            break

    # 出力ディレクトリを確保
    output_dir = ensure_output_dir()

    # アニメーションを保存
    output_path = output_dir / "lissajous.mp4"
    plotter.save_animation(str(output_path))
    plotter.close()


def example_realtime_data():
    """リアルタイムデータプロットの例"""
    print("リアルタイムデータプロットの例")

    # with文を使用した例
    with Plotter(
        x_range=(0, 100),
        y_range=(-10, 10),
        title="リアルタイムデータ",
        color="g",
        use_threading=True,
    ) as plotter:
        plotter.set_xlabel("サンプル")
        plotter.set_ylabel("値")

        # データを初期化
        data_points = 100
        x = np.arange(data_points)
        y = np.zeros(data_points)

        # 100回のデータ更新
        for i in range(100):
            # 新しいデータを生成（ランダムウォーク）
            new_value = y[-1] + np.random.normal(0, 0.5)

            # データシフト
            y = np.roll(y, -1)
            y[-1] = new_value

            # プロット
            key = plotter.plot(x, y, wait_time=50)  # 50ms待機
            if key == 27:  # ESCキーで終了
                break

            # 実際のリアルタイムデータ処理をシミュレート
            time.sleep(0.05)

        # 出力ディレクトリを確保
        output_dir = ensure_output_dir()

        # アニメーションを保存
        output_path = output_dir / "realtime_data.mp4"
        plotter.save_animation(str(output_path))


def example_math_visualization():
    """複雑な数学関数の可視化例"""
    print("複雑な数学関数の可視化例")

    # 3D関数の2Dアニメーション
    plotter = Plotter(
        x_range=(-3, 3),
        y_range=(-3, 3),
        title="関数の等高線",
        figsize=(10, 8),
        color="b",
        line_style=".",  # 点プロット
        max_frames=50,  # 最大50フレーム
    )

    def func(x, y, t):
        # 時間変化する複雑な関数
        return np.sin(x**2 + y**2 + t) * np.exp(-(x**2 + y**2) / 4)

    # 各時刻でZ=0の等高線を抽出
    for t in np.linspace(0, 2 * np.pi, 50):
        points = []
        resolution = 100
        for i in np.linspace(-3, 3, resolution):
            for j in np.linspace(-3, 3, resolution):
                z = func(i, j, t)
                if abs(z) < 0.05:  # Z=0に近い点
                    points.append((i, j))

        if points:
            points = np.array(points)
            x_points, y_points = points[:, 0], y_points = points[:, 1]
            key = plotter.plot(x_points, y_points)
            if key == 27:
                break

    # 出力ディレクトリを確保
    output_dir = ensure_output_dir()

    # アニメーションを保存
    output_path = output_dir / "math_function.mp4"
    plotter.save_animation(str(output_path))
    plotter.close()


def example_simple_with_statement():
    """with文を使ったシンプルな例"""
    print("with文を使ったシンプルな例")

    with Plotter((0, 10), (-1, 1), title="サイン波") as plt:
        plt.set_xlabel("時間 (秒)")
        plt.set_ylabel("振幅")
        for i in range(50):
            x = np.linspace(0, 10, 1000)
            y = np.sin(x + i / 10)
            if plt.plot(x, y) == 27:  # ESCキーで終了
                break

        # 出力ディレクトリを確保
        output_dir = ensure_output_dir()

        # アニメーションを保存
        output_path = output_dir / "simple_sine.mp4"
        plt.save_animation(str(output_path))


def example_stacked_graph():
    """積み上げグラフの表示例"""
    print("積み上げグラフの表示例")

    # サンプルデータを作成（年度ごとの部門別売上データ）
    data = {
        "製品A": [120, 140, 160, 185, 200],
        "製品B": [80, 95, 110, 125, 140],
        "製品C": [60, 70, 85, 95, 110],
        "その他": [30, 35, 40, 45, 50],
    }
    index = ["2019年", "2020年", "2021年", "2022年", "2023年"]
    df = pd.DataFrame(data, index=index)

    # 積み上げグラフを作成
    graph = StackedGraph(
        df,
        figsize=(10, 6),
        title="年度別製品売上（単位：百万円）",
        window_name="Stacked Graph Example",
    )

    # 凡例を追加
    graph.add_legend(loc="upper right")

    # グラフを表示（ESCキーで終了）
    key = graph.show()
    while key != 27:  # ESCキーで終了
        # アニメーションのため一時停止
        time.sleep(0.5)
        key = graph.show()

    # 出力ディレクトリを確保
    output_dir = ensure_output_dir()

    # 静止画を保存
    image_path = output_dir / "stacked_graph_static.png"
    graph.save_image(str(image_path))

    # リソースを解放
    graph.cleanup()


def example_bar_race():
    """バーレースアニメーションの例"""
    print("バーレースアニメーションの例")

    # サンプルデータを作成（各年度の企業売上データ）
    data = {
        "企業A": [100, 120, 140, 180, 210],
        "企業B": [150, 145, 160, 170, 190],
        "企業C": [90, 115, 125, 150, 170],
        "企業D": [80, 100, 120, 130, 140],
        "企業E": [120, 110, 105, 120, 135],
        "企業F": [60, 70, 100, 110, 120],
        "企業G": [70, 90, 95, 100, 110],
    }
    labels = ["2019年", "2020年", "2021年", "2022年", "2023年"]

    # バーレースを作成
    bar_race = BarRace(
        data=data,
        labels=labels,
        title="年度別企業売上ランキング",
        interval=800,  # 800ミリ秒間隔
        top_n=5,  # 上位5社のみ表示
        colormap="tab10",
        figsize=(12, 8),
        xlabel="売上高（億円）",
        ylabel="企業名",
    )

    # アニメーションを作成して表示
    bar_race.animate()
    bar_race.show()

    # 出力ディレクトリを確保
    output_dir = ensure_output_dir()

    # アニメーションを保存
    output_path = output_dir / "bar_race.mp4"
    bar_race.save(str(output_path), fps=2)  # 低フレームレートで保存


def example_broken_barh():
    """不連続な水平バーの例"""
    print("不連続な水平バーの例")

    # プロジェクトスケジュールのサンプルデータ
    broken_barh = BrokenBarh(
        figsize=(12, 6),
        title="プロジェクト進行スケジュール",
        xlabel="日数",
        ylabel="タスク",
        ylabels=["要件定義", "設計", "開発", "テスト", "デプロイ"],
        colormap="Set3",
    )

    # 各タスクのイベントを追加
    broken_barh.add_events([(0, 20)], (0, 0.8), label="要件定義", color="skyblue")
    broken_barh.add_events([(15, 30)], (1, 0.8), label="基本設計", color="lightgreen")
    broken_barh.add_events([(40, 20)], (1, 0.8), label="詳細設計", color="yellowgreen")
    broken_barh.add_events([(30, 40), (75, 10)], (2, 0.8), label="開発", color="salmon")
    broken_barh.add_events([(60, 30), (95, 15)], (3, 0.8), label="テスト", color="violet")
    broken_barh.add_events([(110, 10)], (4, 0.8), label="デプロイ", color="gold")

    # プロットを表示
    broken_barh.plot()
    broken_barh.show()

    # 出力ディレクトリを確保
    output_dir = ensure_output_dir()

    # 画像を保存
    output_path = output_dir / "broken_barh.png"
    broken_barh.save(str(output_path), dpi=300)


if __name__ == "__main__":
    print("Matplotlibアニメーション表示クラスの使用例")
    print("ESCキーを押すと途中で終了します\n")

    # example_sine_wave()
    # example_lissajous()
    # example_realtime_data()
    # example_math_visualization()
    # example_simple_with_statement()
    # example_stacked_graph()
    example_bar_race()
    # example_broken_barh()

    print("\nすべての例が終了しました")
