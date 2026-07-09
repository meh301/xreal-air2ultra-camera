// preview_clean.swift — XREAL Air 2 Ultra トラッキングカメラのリアルタイム・クリーンプレビュー
//
// UVCストリーム(640x482, 実体は8bitモノクロ)を取得し、フレーム毎に
//   1. ブロックスクランブル解除 (128ブロック x 2400バイト, descramble/xreal_descramble.py と同一アルゴリズム)
//   2. 列FPN(縦縞)除去: 水平ハイパスの列メディアンをEMAで蓄積して減算
//   3. 行バンディング除去 + ヒストグラム均等化
// を行い、左右カメラを並べて表示する (480x640 x2 = 960x640)。
//
// Build & run:
//   swiftc -O preview_clean.swift -o preview_clean -framework AVFoundation -framework AppKit
//   ./preview_clean                      # GUIプレビュー
//   ./preview_clean --snap out.png       # ウィンドウなしで数フレーム処理してスナップ保存(動作確認用)
//   ./preview_clean --test in.pgm outpfx # 単一PGMをデスクランブルして保存(検証用)
// Keys:  c = clean/scrambled 切替   s = スナップ保存   space = pause   q/esc = quit
//
import AppKit
import AVFoundation

// 要・最新ファームウェア (MCU 12.1.00.498_20241115)。古い場合は
// https://ota.xreal.com/ultra-update?version=1 でアップデートしてください。
let W = 640, HIMG = 480, HFULL = 482, META_ROW = 480, CTR_COL = 18
// row480 col59 はカメラ識別ビット: 1 = cam0 (is_right=True), 0 = cam1 (is_right=False)。
// ペア内の到着順は固定でないため、順序ベースの判定は使わないこと。
let CAM_COL = 59
let NB = 128, BS = 2400          // 128 blocks x 2400 bytes = 307200 = 640*480
let OW = 480, OH = 640           // デスクランブル後は縦向き 480x640

let REORDER: [Int] = [
    119, 54, 21, 0, 108, 22, 51, 63, 93, 99, 67, 7, 32, 112, 52, 43,
    14, 35, 75, 116, 64, 71, 44, 89, 18, 88, 26, 61, 70, 56, 90, 79,
    87, 120, 81, 101, 121, 17, 72, 31, 53, 124, 127, 113, 111, 36, 48,
    19, 37, 83, 126, 74, 109, 5, 84, 41, 76, 30, 110, 29, 12, 115, 28,
    102, 105, 62, 103, 20, 3, 68, 49, 77, 117, 125, 106, 60, 69, 98, 9,
    16, 78, 47, 40, 2, 118, 34, 13, 50, 46, 80, 85, 66, 42, 123, 122,
    96, 11, 25, 97, 39, 6, 86, 1, 8, 82, 92, 59, 104, 24, 15, 73, 65,
    38, 58, 10, 23, 33, 55, 57, 107, 100, 94, 27, 95, 45, 91, 4, 114,
]

// ---- descramble -------------------------------------------------------------
func buildLUT(isRight: Bool) -> [Int32] {
    var lut = [Int32](repeating: 0, count: NB * BS)
    var pY = 0, pX = 0, idx = 0
    for _ in 0..<NB {
        var off = 0
        while off < BS {
            let seg = min(pY + (BS - off), 640) - pY
            for k in 0..<seg {
                let r = pX, c = pY + k
                let y = isRight ? c : 639 - c
                let x = isRight ? r : 479 - r
                lut[idx] = Int32(y * 480 + x)
                idx += 1
            }
            off += seg
            pY += seg
            if pY >= 640 { pX += 1; pY = 0 }
        }
    }
    return lut
}

func descramble(_ raw: UnsafePointer<UInt8>, lut: [Int32]) -> [UInt8] {
    // 同期検出: 先頭128バイトの合計が最小のブロックが REORDER[align]
    var minScore = Int.max, minBlock = 0
    for b in 0..<NB {
        var s = 0
        for k in 0..<128 { s += Int(raw[b * BS + k]) }
        if s < minScore { minScore = s; minBlock = b }
    }
    let align = REORDER.firstIndex(of: minBlock)!
    var out = [UInt8](repeating: 0, count: NB * BS)
    out.withUnsafeMutableBufferPointer { o in
        lut.withUnsafeBufferPointer { l in
            for t in 0..<NB {
                let src = raw + REORDER[(align + t) % NB] * BS
                let lbase = t * BS
                for k in 0..<BS { o[Int(l[lbase + k])] = src[k] }
            }
        }
    }
    return out
}

// ---- image processing -------------------------------------------------------
func equalize(_ src: [UInt8]) -> [UInt8] {
    var hist = [Int](repeating: 0, count: 256)
    for v in src { hist[Int(v)] += 1 }
    var cdf = [Int](repeating: 0, count: 256); var acc = 0
    for i in 0..<256 { acc += hist[i]; cdf[i] = acc }
    var cdfmin = 0
    for i in 0..<256 where cdf[i] != 0 { cdfmin = cdf[i]; break }
    let denom = max(1, src.count - cdfmin)
    var lut = [UInt8](repeating: 0, count: 256)
    for i in 0..<256 { lut[i] = UInt8(max(0, min(255, (cdf[i] - cdfmin) * 255 / denom))) }
    var out = [UInt8](repeating: 0, count: src.count)
    for i in 0..<src.count { out[i] = lut[Int(src[i])] }
    return out
}

/// カメラ毎の縦縞FPN推定と除去(process_clean.py と同じ考え方のオンライン版)
final class Cleaner {
    var stripe = [Float](repeating: 0, count: OW)
    var haveStripe = false
    private var f = [Float](repeating: 0, count: OW * OH)
    private var hp = [Float](repeating: 0, count: OW * OH)

    func clean(_ img: [UInt8]) -> [UInt8] {
        for i in 0..<OW * OH { f[i] = Float(img[i]) }
        let r = 15

        // 列FPN: 水平ハイパス → 列メディアン → EMA蓄積 → 減算
        var pref = [Float](repeating: 0, count: OW + 1)
        for y in 0..<OH {
            let base = y * OW
            for x in 0..<OW { pref[x + 1] = pref[x] + f[base + x] }
            for x in 0..<OW {
                let lo = max(0, x - r), hi = min(OW - 1, x + r)
                hp[base + x] = f[base + x] - (pref[hi + 1] - pref[lo]) / Float(hi - lo + 1)
            }
        }
        var cur = [Float](repeating: 0, count: OW)
        var colbuf = [Float](repeating: 0, count: OH)
        for x in 0..<OW {
            for y in 0..<OH { colbuf[y] = hp[y * OW + x] }
            colbuf.sort()
            cur[x] = colbuf[OH / 2]
        }
        if !haveStripe {
            stripe = cur; haveStripe = true
        } else {
            for x in 0..<OW { stripe[x] = 0.95 * stripe[x] + 0.05 * cur[x] }
        }
        for y in 0..<OH {
            let base = y * OW
            for x in 0..<OW { f[base + x] -= stripe[x] }
        }

        // 行バンディング: 垂直ハイパス → 行メディアン → 減算 (時間変動するのでEMAなし)
        var cpref = [Float](repeating: 0, count: OH + 1)
        for x in 0..<OW {
            for y in 0..<OH { cpref[y + 1] = cpref[y] + f[y * OW + x] }
            for y in 0..<OH {
                let lo = max(0, y - r), hi = min(OH - 1, y + r)
                hp[y * OW + x] = f[y * OW + x] - (cpref[hi + 1] - cpref[lo]) / Float(hi - lo + 1)
            }
        }
        var rowbuf = [Float](repeating: 0, count: OW)
        for y in 0..<OH {
            let base = y * OW
            for x in 0..<OW { rowbuf[x] = hp[base + x] }
            rowbuf.sort()
            let m = rowbuf[OW / 2]
            for x in 0..<OW { f[base + x] -= m }
        }

        var out = [UInt8](repeating: 0, count: OW * OH)
        for i in 0..<OW * OH { out[i] = UInt8(max(0, min(255, f[i]))) }
        return equalize(out)
    }
}

// ---- I/O helpers --------------------------------------------------------------
func cgGray(_ pixels: [UInt8], _ w: Int, _ h: Int) -> CGImage? {
    guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
    return CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: w,
                   space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: 0),
                   provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
}

func savePNG(_ pixels: [UInt8], _ w: Int, _ h: Int, _ path: String) {
    guard let cg = cgGray(pixels, w, h) else { return }
    let rep = NSBitmapImageRep(cgImage: cg)
    try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: path))
}

func readPGM(_ path: String) -> [UInt8]? {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else { return nil }
    // "P5\n<w> <h>\n255\n" の3行ヘッダ前提 (xreal_cam の出力)
    var idx = 0, newlines = 0
    while idx < data.count && newlines < 3 {
        if data[idx] == 0x0a { newlines += 1 }
        idx += 1
    }
    return [UInt8](data[idx...])
}

// ---- camera -----------------------------------------------------------------
final class FrameView: NSView {
    var image: CGImage?
    var label = ""
    override var isFlipped: Bool { true }
    override func draw(_ dirtyRect: NSRect) {
        NSColor.black.setFill(); dirtyRect.fill()
        guard let img = image, let ctx = NSGraphicsContext.current?.cgContext else { return }
        ctx.interpolationQuality = .none
        ctx.draw(img, in: bounds)
        let attrs: [NSAttributedString.Key: Any] = [
            .foregroundColor: NSColor.green,
            .font: NSFont.monospacedSystemFont(ofSize: 13, weight: .bold)]
        label.draw(at: NSPoint(x: 8, y: 6), withAttributes: attrs)
    }
}

final class Cam: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    weak var view: FrameView?
    var showClean = true
    var paused = false
    var snapRequest = false
    var snapPath: String?            // --snap モード: 保存したら終了
    let sem = DispatchSemaphore(value: 0)

    private var snapCount = 0
    private let luts = [buildLUT(isRight: true), buildLUT(isRight: false)]  // cam0, cam1
    private let cleaners = [Cleaner(), Cleaner()]
    private var latestClean: [[UInt8]?] = [nil, nil]
    private var latestRaw: [[UInt8]?] = [nil, nil]
    private var fpsCount = 0
    private var fpsStart = Date()
    private var fps = 0.0

    func start() {
        guard let dev = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.external, .continuityCamera], mediaType: .video, position: .unspecified)
            .devices.first(where: { $0.localizedName.contains("UVC Camera") || $0.manufacturer.contains("XREAL") }) else {
            FileHandle.standardError.write(Data("XREAL UVC camera not found.\n".utf8))
            exit(1)
        }
        print("Opening \(dev.localizedName) [\(dev.modelID)]")
        guard let input = try? AVCaptureDeviceInput(device: dev) else { exit(1) }
        session.beginConfiguration()
        if session.canAddInput(input) { session.addInput(input) }
        let out = AVCaptureVideoDataOutput()
        out.setSampleBufferDelegate(self, queue: DispatchQueue(label: "xreal.clean"))
        if session.canAddOutput(out) { session.addOutput(out) }
        session.commitConfiguration()
        session.startRunning()
    }

    func captureOutput(_ o: AVCaptureOutput, didOutput s: CMSampleBuffer, from c: AVCaptureConnection) {
        if paused { return }
        guard let px = CMSampleBufferGetImageBuffer(s) else { return }
        CVPixelBufferLockBaseAddress(px, .readOnly)
        let bpr = CVPixelBufferGetBytesPerRow(px)
        let bufH = CVPixelBufferGetHeight(px)
        guard let base = CVPixelBufferGetBaseAddress(px) else {
            CVPixelBufferUnlockBaseAddress(px, .readOnly); return
        }
        let p = base.assumingMemoryBound(to: UInt8.self)
        var mono = [UInt8](repeating: 0, count: W * HFULL)
        let line = 2 * W
        mono.withUnsafeMutableBufferPointer { d in
            for y in 0..<bufH { memcpy(d.baseAddress! + y * line, p + y * bpr, line) }
        }
        CVPixelBufferUnlockBaseAddress(px, .readOnly)

        var sum = 0
        for i in stride(from: 0, to: W * HIMG, by: 7) { sum += Int(mono[i]) }
        if sum / (W * HIMG / 7) < 5 { return }                    // 起動直後の黒フレームは無視

        let counter = Int(mono[META_ROW * W + CTR_COL])
        let cam = 1 - (Int(mono[META_ROW * W + CAM_COL]) & 1)     // テレメトリのカメラ識別ビットで確実に判定

        mono.withUnsafeBufferPointer { buf in
            latestClean[cam] = cleaners[cam].clean(descramble(buf.baseAddress!, lut: luts[cam]))
        }
        latestRaw[cam] = equalize(Array(mono[0..<(W * HIMG)]))

        fpsCount += 1
        let dt = Date().timeIntervalSince(fpsStart)
        if dt >= 1 { fps = Double(fpsCount) / dt; fpsCount = 0; fpsStart = Date() }

        // 両カメラ揃っていれば毎フレーム表示を更新(到着順に依存しない)
        guard let c0 = latestClean[0], let c1 = latestClean[1],
              let r0 = latestRaw[0], let r1 = latestRaw[1] else { return }

        let cw = showClean ? 2 * OW : 2 * W
        let ch = showClean ? OH : HIMG
        var combined = [UInt8](repeating: 0, count: cw * ch)
        let (a, b, sw) = showClean ? (c0, c1, OW) : (r0, r1, W)
        for y in 0..<ch {
            for x in 0..<sw {
                combined[y * cw + x] = a[y * sw + x]
                combined[y * cw + sw + x] = b[y * sw + x]
            }
        }

        if let path = snapPath {
            snapCount += 1
            if snapCount >= 15 {                                  // FPNのEMAが安定してから保存
                savePNG(combined, cw, ch, path)
                print("snapshot saved: \(path)")
                session.stopRunning()
                sem.signal()
            }
            return
        }
        if snapRequest {
            snapRequest = false
            let path = "preview_snap_\(Int(Date().timeIntervalSince1970)).png"
            savePNG(combined, cw, ch, path)
            print("snapshot saved: \(path)")
        }

        guard let cg = cgGray(combined, cw, ch) else { return }
        let lbl = String(format: "XREAL Air 2 Ultra  L | R  ctr=%d  %.0f fps  [%@]  (c:view s:snap space:pause q:quit)",
                         counter, fps, showClean ? "CLEAN" : "SCRAMBLED")
        DispatchQueue.main.async {
            self.view?.image = cg
            self.view?.label = lbl
            self.view?.needsDisplay = true
        }
    }
}

// ---- entry ------------------------------------------------------------------
let args = CommandLine.arguments

if let i = args.firstIndex(of: "--test"), args.count > i + 2 {
    guard let raw = readPGM(args[i + 1]), raw.count >= NB * BS else {
        print("cannot read \(args[i + 1])"); exit(1)
    }
    for (name, isRight) in [("right", true), ("left", false)] {
        let lut = buildLUT(isRight: isRight)
        raw.withUnsafeBufferPointer { buf in
            let img = descramble(buf.baseAddress!, lut: lut)
            savePNG(img, OW, OH, "\(args[i + 2])_\(name).png")
        }
    }
    print("saved \(args[i + 2])_right.png / _left.png")
    exit(0)
}

let cam = Cam()

if let i = args.firstIndex(of: "--snap"), args.count > i + 1 {
    cam.snapPath = args[i + 1]
    cam.start()
    if cam.sem.wait(timeout: .now() + 20) == .timedOut {
        print("snap timed out"); exit(1)
    }
    exit(0)
}

let app = NSApplication.shared
final class AppDelegate: NSObject, NSApplicationDelegate {
    var win: NSWindow!
    var view: FrameView!
    func applicationDidFinishLaunching(_ n: Notification) {
        let w = 2 * OW, h = OH
        win = NSWindow(contentRect: NSRect(x: 100, y: 100, width: w, height: h),
                       styleMask: [.titled, .closable, .miniaturizable, .resizable],
                       backing: .buffered, defer: false)
        win.title = "XREAL Air 2 Ultra — clean stereo preview"
        win.contentAspectRatio = NSSize(width: w, height: h)
        view = FrameView(frame: NSRect(x: 0, y: 0, width: w, height: h))
        win.contentView = view
        win.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        cam.view = view
        cam.start()
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ s: NSApplication) -> Bool { true }
}
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
NSEvent.addLocalMonitorForEvents(matching: .keyDown) { e in
    switch e.charactersIgnoringModifiers {
    case "c": cam.showClean.toggle()
    case "s": cam.snapRequest = true
    case " ": cam.paused.toggle()
    case "q", "\u{1b}": NSApp.terminate(nil)
    default: break
    }
    return e
}
app.run()
