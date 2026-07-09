# XREAL Air 2 Ultra — ステレオカメラビューア

**XREAL Air 2 Ultra のステレオトラッキングカメラを macOS / Windows / Linux / Android で読み取る — SDK不要、kext不要、ドライバ不要。**

[English README is here](README.md)

Air 2 Ultra はトラッキングカメラを標準の UVC (USB Video Class) デバイスとして公開して
いますが、映像ストリームは**ブロック単位でスクランブル**されており、通常のWebカメラ
ビューアではノイズにしか見えません。本プロジェクトはこれをリアルタイムでデスクランブルし、
クリーンな 640×480 ステレオグレースケール映像を取得します — VIO/SLAM実験、ロボティクス、
ハードウェア解析などの入力にそのまま使えます。

![clean stereo preview](docs/images/stereo_preview.png)

## クイックスタート

### Windows / Linux (macOSでも可)

必要なもの: Python 3.9+ と `numpy`、`opencv-python`、USB-C接続した Air 2 Ultra。
Windowsでは `ffmpeg` もPATHに必要です(`winget install ffmpeg`、下記の注意参照)。

```sh
pip install numpy opencv-python
python python/preview_clean.py
```

左右カメラのライブ映像(デスクランブル+ノイズ除去済み)のウィンドウが、UVCの
フルレート60fpsで開きます。XREALのストリームはテレメトリ行のフィンガープリントで
自動検出されるため、他のWebカメラが混ざっていても問題ありません。
`python python/xreal_uvc.py` で検出状況を確認できます。

キー操作: `c` クリーン/スクランブル表示切替 · `s` スナップショットPNG保存 · `space` 一時停止 · `q` 終了

クリーンなステレオペアは毎フレーム**名前付き共有メモリのフレームバッファ**
(`xreal_stereo_fb`、960×640グレースケール+seqlockヘッダ)にも書き込まれるため、
Python / C++ / Rust / Unity など任意のプロセスからソケット不要・約1msの遅延で
ライブ映像を利用できます。`python python/xreal_fb.py --show` がデモコンシューマで、
64バイトヘッダのレイアウトは [python/xreal_fb.py](python/xreal_fb.py) に記載しています。
`--headless` でウィンドウなしのパブリッシャとして動きます。

プラットフォーム別の注意:
- **Windows**: ffmpeg をインストールしてください(`winget install ffmpeg`)。
  OpenCVのWindowsバックエンドはこのデバイスの生ストリームを取得できないことが
  実測で判明しているため(MSMFはストリーム開始不能、DirectShowは強制BGR変換)、
  ツールは自動的に ffmpeg の dshow 入力(ネイティブピンフォーマット取得)経由で
  キャプチャします。
- **Linux**: `/dev/video*` へのアクセス権が必要です(通常は `video` グループ:
  `sudo usermod -aG video $USER` 後に再ログイン)。グラスは通常の `uvcvideo`
  デバイスとして見えるので、閲覧だけなら udev ルールは不要です。

### macOS (ネイティブツール)

必要なもの: Xcode Command Line Tools (`xcode-select --install`)。

```sh
make
./preview_clean
```

初回実行時にターミナルへのカメラ権限を求められます。

### Android

Android には外付けUVCデバイス用のカメラAPIがないため、[`android/`](android/) に
USBホストAPI経由でグラスを開くネイティブアプリ(libusb + libuvc + C実装の
デスクランブラ)を用意しています。Android Studio または Gradle でビルドできます:

```sh
cd android
./fetch_deps.sh        # Windowsでは fetch_deps.ps1 — libusb/libuvc をピン留めタグで取得
./gradlew :app:assembleDebug
```

APKをインストールし、グラスをスマホのUSB-Cポートに接続してUSB権限のダイアログを
許可すると、ライブステレオプレビューが始まります。詳細は
[android/README.md](android/README.md) を参照してください。

## ツール一覧

| ツール | 機能 |
|--------|------|
| `python/preview_clean.py` | クロスプラットフォームのリアルタイムステレオビューア(デスクランブル+固定パターンノイズ除去)。`--snap out.png` でウィンドウなしスナップショット、`--test in.pgm prefix` でオフライン検証。 |
| `python/xreal_cam.py` | クロスプラットフォームのレコーダー: `python xreal_cam.py <フレーム数> <出力dir>` で生の `cam0_*.pgm` / `cam1_*.pgm`(スクランブルされたまま)と `meta.csv` を保存。macOS版レコーダーと同一形式。 |
| `python/xreal_uvc.py` | 上記2ツールが使うキャプチャモジュール(バックエンド探索、ffmpegフォールバック、テレメトリによる自動検出、バイト順補正)。直接実行するとXREALストリームをスキャン。 |
| `python/xreal_fb.py` | 共有メモリフレームバッファ: レイアウト定義、Writer/Readerクラス、デモコンシューマ(`python xreal_fb.py --show`)。 |
| `python/xreal_imu.py` | **1kHz IMUリーダー**(要 `pip install hidapi`): ライブ表示、`--csv` ログ、`--fb` 共有メモリリング、`--config calib.json` でデバイス内の工場キャリブレーション(IMUバイアス/ノイズ+両トラッキングカメラの魚眼内部パラメータとカメラ↔IMU外部パラメータ — VIOに必要な全て)をダンプ。 |
| `preview_clean` (macOS) | ビューアのネイティブSwift版、60fps。キー・フラグは共通。 |
| `xreal_cam` (macOS) | レコーダーのネイティブSwift版。 |
| `enumerate` (macOS) | AVFoundationのカメラ一覧とXREALデバイスのフォーマットを表示。 |
| `android/` | Androidアプリ: USBホストAPI + libusb/libuvc、C実装のデスクランブル+ノイズ除去、左右並置ライブプレビューとスナップショット。 |
| `python/process_clean.py` | オフラインパイプライン: `python3 python/process_clean.py <キャプチャdir> <出力dir>` で録画(どちらのレコーダーの出力でも可)をデスクランブル+クリーン化してPNGと左右並置の `stereo_feed.mp4` を生成。 |
| `python/xreal_descramble.py` | 単一フレーム用の最小デスクランブラ。リファレンス実装。 |
| `research/` | リバースエンジニアリング用ツール群(ベンダーHIDコマンド、UVCコントロール、USBディスクリプタ)。大半はmacOSとLinuxでビルド可能。[research/README.md](research/README.md) 参照。 |

## 仕組み(概要)

- グラスは通常のUVC Webカメラとして列挙されます(`640×482 @ 60fps`、名目上YUVですが
  実際は**1バイト=1モノクロ画素**)。0〜479行目が画像、480行目がテレメトリ、481行目がパディング。
- 各フレームの画像307,200バイトは**128ブロック×2,400バイト**の固定順列でシャッフルされ、
  開始位相がフレーム毎に回転します。魚眼レンズの黒い縁で始まるブロックを探すことで同期を回復します。
- 連続するUVCフレームは左右カメラを交互に運びますが、**順序は固定ではありません** —
  テレメトリ行の58バイト目(`0x20`/`0x21`)がどちらのカメラかを示します。
- センサーは90°回転(かつ左右で互いに180°逆向き)に実装されているため、デスクランブラは
  回転も行います。出力は片眼あたり480×640の縦向きです。
- 残る縦縞(列固定パターンノイズ)はオンラインで推定して減算します。
- YUVラベルが偽物なので、UVCスタックによっては16bitペアの2バイトが入れ替わって届きます。
  テレメトリのマーカーでこれを検出でき、本リポジトリの全キャプチャ経路が自動補正します。

グラスはベンダーHIDインターフェース経由で**1kHzのIMU**もストリームしており、
工場キャリブレーション一式(両トラッキングカメラの魚眼内部パラメータと
カメラ↔IMU外部パラメータ、IMUバイアス・ノイズ密度)をJSONとしてデバイス内に
保持しています — どちらも `python/xreal_imu.py` で読み出せます。

プロトコルの詳細(USBレイアウト、テレメトリ行のマップ、スクランブルアルゴリズム、
IMUパケットフォーマットとキャリブレーションブロブ、UVC露出コントロール、
ベンダーHIDプロトコル、OS別キャプチャノート): **[docs/PROTOCOL.md](docs/PROTOCOL.md)**

## クレジット

- ブロック並べ替え表は [mazeasdamien/myXreal](https://github.com/mazeasdamien/myXreal)
  (`stereo_camera.cpp`)の解析成果です。
- ベンダーHIDパケットフォーマットは
  [badicsalex/ar-drivers-rs](https://github.com/badicsalex/ar-drivers-rs) に文書化されています。
- Androidアプリは [libusb](https://github.com/libusb/libusb)(LGPL-2.1、独立した
  共有ライブラリとして分離)と [libuvc](https://github.com/libuvc/libuvc)(BSD)を利用しています。
- 本リポジトリの解析・ツール・ドキュメントは [Claude Code](https://claude.com/claude-code)
  (Claude Fable 5) との協働で開発されました。

## 免責事項

本プロジェクトは非公式のリバースエンジニアリング成果であり、XREALとは無関係です。
標準UVC経由でカメラストリームを*読み取るだけ*で、デバイスを変更するコマンドは送信しませんが、
使用は自己責任でお願いします。

## ライセンス

[MIT](LICENSE)
