# XREAL Air 2 Ultra — ステレオカメラ & IMU、SDK不要

**Air 2 Ultra のセンサーが持つすべてを Windows / Linux / macOS / Android で:
リアルタイムにデスクランブルした2つのトラッキングカメラ、1kHz IMU、
デバイス内蔵の工場キャリブレーション — SDK不要、kext不要、ドライバ不要。**

[English README is here](README.md)

グラスはステレオトラッキングカメラを標準UVC Webカメラとして公開していますが、
映像は**ブロック単位でスクランブル**されており通常のビューアではノイズにしか
見えません。本プロジェクトはこれをライブでクリーンな 640×480 ステレオ
グレースケール映像にデスクランブルし、同じナノ秒クロックを共有するIMUを読み、
魚眼内部パラメータとカメラ↔IMU外部パラメータをデバイスから取り出します —
VIO/SLAM・ロボティクス・ハードウェア解析のための完全な生入力です。

![clean stereo preview](docs/images/stereo_preview.png)

## クイックスタート

> [!IMPORTANT]
> **まずグラスをアップデートしてください。** 本リポジトリの全ツールは最新の
> グラスファームウェア(MCU `12.1.00.498_20241115`)が必要です — 古い
> ファームウェアはストリームのメタデータ形式が異なり、デコードできません。
> アップデートはブラウザから1分で完了します:
> **<https://ota.xreal.com/ultra-update?version=1>**(古いファームウェアを
> 検出した場合はツール側でもこの案内を表示します。現在のバージョンは
> `python python/xreal_imu.py --info` で確認できます)。

### Windows / Linux / macOS (Python)

```sh
pip install numpy opencv-python hidapi
python python/preview_clean.py           # ライブステレオビューア、60fps
python python/xreal_imu_scope.py         # IMUオシロスコープ+3D姿勢
python python/glasses_passthrough.py     # カメラ映像をグラスへ(SBS)
```

ビューアのキー: `c` クリーン/スクランブル切替 · `s` スナップショット ·
`space` 一時停止 · `q` 終了。XREALはテレメトリのフィンガープリントで
自動検出されます。Windowsでは ffmpeg も必要(`winget install ffmpeg`)、
LinuxのIMUにはudevルールが必要 — どちらも
[プラットフォーム別の注意](#プラットフォーム別の注意) を参照。

### macOS (ネイティブツール)

```sh
xcode-select --install   # 初回のみ
make
./preview_clean          # ステレオビューア(キー/フラグはPython版と共通)
./xreal_imu              # IMUリーダー (--csv / --config / --info)
./glasses_passthrough    # カメラ映像をグラス自身のディスプレイへ(SBS)
```

初回実行時にターミナルへのカメラ権限を求められます。上記のPythonツールも
macOSで動作します。

### Android

AndroidのカメラAPIは外付けUVCデバイスに届かないため、[`android/`](android/)
はUSBホストAPI上のネイティブアプリです(libusb + libuvc + C実装デスクランブラ):

```sh
cd android
./fetch_deps.sh          # Windowsでは fetch_deps.ps1 — libusb/libuvcをピン留めタグで取得
./gradlew :app:assembleDebug
```

APKをインストールしてグラスをスマホに接続すると、ライブステレオプレビューと
1kHz IMU表示(フュージョンした姿勢つき)が始まります。詳細:
[android/README.md](android/README.md)。

## ツール

**カメラ**

| コマンド | 機能 |
|----------|------|
| `python python/preview_clean.py` | リアルタイムステレオビューア。クリーンな各ペアを共有メモリフレームバッファにも書き込む。`--headless`(ウィンドウなし)、`--snap out.png`(スナップショット1枚)、`--test in.pgm pfx`(オフライン検証)。 |
| `python python/glasses_passthrough.py` | カメラ映像をグラス自身のディスプレイへステレオパススルー(左カメラ→左目。グラスを3D/SBSモードに)。`--list`/`--display N` でディスプレイ選択、`--window` でグラスなしプレビュー、`--geometry X,Y,W,H` で手動配置。キー: `x` 入替 · `r` 回転 · `m` ミラー · `s` SBS。 |
| `python python/xreal_cam.py <N> <dir>` | 生(スクランブルのまま)フレームをN枚 `cam{0,1}_*.pgm` + `meta.csv` として録画。macOS版レコーダーと同一形式。 |
| `python python/process_clean.py <dir> <out>` | オフラインパイプライン: 録画をデスクランブル+ノイズ除去してPNGと左右並置の `stereo_feed.mp4` に。 |
| `python python/xreal_uvc.py` | キャプチャバックエンドのスキャン/診断(このファイルは各ツールが使うキャプチャライブラリでもある)。 |

**IMU & キャリブレーション**

| コマンド | 機能 |
|----------|------|
| `python python/xreal_imu.py` | 1kHzジャイロ/加速度リーダー。`--quat` ホスト側クォータニオン、`--csv` 全サンプル記録、`--fb` 共有メモリリング、`--config c.json` 工場キャリブレーションのダンプ、`--info` シリアル+ファームウェアバージョン。 |
| `python python/xreal_imu_scope.py` | ローリングオシロスコープ(ジャイロ+加速度)+3D姿勢ビュー。`b` でジャイロバイアス再取得。カメラビューアと並行動作可。 |

**macOSネイティブバイナリ** — `make` で `preview_clean`、`xreal_cam`、
`xreal_imu`、`enumerate`(デバイス/フォーマット一覧)をビルド。
Python版と同等の機能です。さらにmacOS専用ツールが1つ:

| コマンド | 機能 |
|----------|------|
| `./glasses_passthrough` | カメラ映像をグラス自身のディスプレイへステレオパススルー — 左カメラ→左目、右カメラ→右目、XREALディスプレイに左右並置でフルスクリーン表示(グラスを3D/SBSモードに)。`--list` でディスプレイ一覧、`--display N` で選択、`--window` でグラスなしプレビュー。キー: `x` 左右入替 · `r` 回転 · `m` ミラー · `s` SBS切替。 |

**リファレンス & リサーチ** —
[`python/xreal_descramble.py`](python/xreal_descramble.py) は各実装の検証に
使う最小の単一フレームデスクランブラ。[`research/`](research/README.md) には
リバースエンジニアリング用ツール群(ベンダーHIDコマンド、UVCコントロール、
USBディスクリプタ)。

## 自分のコードからデータを使う

USBに触れずにVIO/SLAMパイプラインへ供給するための仕組みです:

- **カメラフレームバッファ** — クリーンな各ステレオペアが名前付き共有メモリ
  `xreal_stereo_fb`(960×640グレースケール、ペアカウンタとデバイス露光
  タイムスタンプ入りのseqlockヘッダ)に書き込まれます。遅延約1ms、任意の
  言語から利用可。デモコンシューマ: `python python/xreal_fb.py --show`。
  64バイトヘッダのレイアウトは [python/xreal_fb.py](python/xreal_fb.py) 参照。
- **IMUリング** — `xreal_imu.py --fb` が全サンプルを `xreal_imu_fb` リングに
  書き込みます(レイアウトは [python/xreal_imu.py](python/xreal_imu.py))。
- **単一クロック** — カメラの露光タイムスタンプとIMUタイムスタンプは同じ
  フリーランニングのナノ秒カウンタです(実機で検証済み。カメラ側は下位
  32bit)。クロック間キャリブレーション不要。アンラップ手順は
  [PROTOCOL.md](docs/PROTOCOL.md#clock-domains-cameras-vs-imu)。
- **工場キャリブレーション** — `xreal_imu.py --config` で両カメラの
  fisheye624内部パラメータ、カメラ↔IMU外部パラメータ、IMUバイアス+ノイズ
  密度をダンプ: そのまま使えるVIOパラメータ一式です。(デバイスのシリアルを
  含むため、公開リポジトリには入れないこと。)
- **姿勢** — デバイスは生の慣性データのみを送信します。クォータニオンは
  ホスト側のMadgwickフィルタ
  ([python/xreal_ahrs.py](python/xreal_ahrs.py))で計算します。

## 仕組み(概要)

- グラスは通常のUVC Webカメラとして列挙されます(`640×482 @ 60fps`、名目上
  YUVですが実際は**1バイト=1モノクロ画素**)。0〜479行目が画像、480行目が
  テレメトリ、481行目がパディング。
- 各フレームの画像307,200バイトは**128ブロック×2,400バイト**の固定順列で
  シャッフルされ、開始位相がフレーム毎に回転します。魚眼レンズの黒い縁で
  始まるブロックを探すことで同期を回復します。
- 連続するUVCフレームは左右カメラを交互に運びますが、**順序は固定では
  ありません** — テレメトリのバイトがどちらのカメラかを示します。
- センサーは90°回転(かつ左右で互いに180°逆向き)に実装されているため、
  デスクランブラは回転も行います。出力は片眼あたり480×640の縦向きです。
- 残る縦縞(列固定パターンノイズ)はオンラインで推定して減算します。
- 各ツールは**現行のグラスファームウェア**(MCU `12.1.00.498`、最新版 —
  <https://ota.xreal.com/ultra-update?version=1> でアップデート)が必要です。
  古いファームウェアはテレメトリのレイアウトが異なり、サポートされません。
  また、UVCスタックによっては偽YUVペアのバイトが入れ替わって届きますが、
  全キャプチャ経路が自動検出・補正します。

プロトコルの完全なドキュメント — USBレイアウト、テレメトリ行、スクランブル
アルゴリズム、IMUパケットフォーマット、キャリブレーションブロブ、クロック
ドメイン、UVC露出コントロール、ベンダーHIDプロトコル、OS別キャプチャノート:
**[docs/PROTOCOL.md](docs/PROTOCOL.md)**

## プラットフォーム別の注意

- **Windows**: ffmpegをインストールしてください(`winget install ffmpeg`)。
  OpenCVのWindowsバックエンドはこのデバイスの生ストリームを取得できません
  (MSMFはストリーム開始不能、DirectShowは強制BGR変換)。そのためツールは
  ネイティブピンフォーマットを取得できる ffmpeg の dshow 入力経由で
  キャプチャします。
- **Linux**: カメラ閲覧には `/dev/video*` へのアクセス権が必要です(通常は
  `video` グループ: `sudo usermod -aG video $USER` 後に再ログイン)。IMUには
  hidrawノードの権限が必要: `sudo cp linux/99-xreal-air2ultra.rules
  /etc/udev/rules.d/ && sudo udevadm control --reload && sudo udevadm trigger`
  の後、挿し直してください。
- **macOS**: ビューア初回実行時のカメラ権限ダイアログを許可してください。

## クレジット

- ブロック並べ替え表は [mazeasdamien/myXreal](https://github.com/mazeasdamien/myXreal)
  (`stereo_camera.cpp`)の解析成果です — Windowsネイティブの姉妹プロジェクト
  (ImGuiダッシュボード、ステレオ平行化、グラス上のVRシーン)としても一見の
  価値があります。
- ベンダーHIDパケットフォーマットは
  [badicsalex/ar-drivers-rs](https://github.com/badicsalex/ar-drivers-rs) と
  [wheaney/nrealAirLinuxDriver](https://gitlab.com/wheaney/nrealAirLinuxDriver)
  に文書化されています。
- Androidアプリは [libusb](https://github.com/libusb/libusb)(LGPL-2.1、独立した
  共有ライブラリとして分離)と [libuvc](https://github.com/libuvc/libuvc)(BSD)を
  利用しています。
- 本リポジトリの解析・ツール・ドキュメントは [Claude Code](https://claude.com/claude-code)
  (Claude Fable 5) との協働で開発されました。

## 免責事項

本プロジェクトは非公式のリバースエンジニアリング成果であり、XREALとは
無関係です。センサーストリームの読み取りと、コミュニティで検証済みの
問い合わせ/有効化コマンドの送信のみを行いますが、使用は自己責任で
お願いします。

## ライセンス

[MIT](LICENSE)
