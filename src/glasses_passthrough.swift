// glasses_passthrough.swift — XREAL Air 2 Ultra のトラッキングカメラ映像を
// グラス本体の左右ディスプレイに表示するステレオ・パススルー。
//
// UVCストリーム(640x482, 実体は8bitモノクロ)を取得し、フレーム毎に
//   1. ブロックスクランブル解除 (128ブロック x 2400バイト, preview_clean.swift と同一)
//   2. 列FPN(縦縞)除去 + 行バンディング除去 + ヒストグラム均等化
// を行い、左カメラを画面左半分・右カメラを画面右半分に並べて(SBS)フルスクリーン表示する。
// グラスを 3D(Side-by-Side)モードにすると、左半分が左目・右半分が右目に写る。
//
// Build:
//   swiftc -O glasses_passthrough.swift -o glasses_passthrough -framework AVFoundation -framework AppKit
// Run:
//   ./glasses_passthrough              # グラスのディスプレイを自動検出しフルスクリーン表示
//   ./glasses_passthrough --list       # 接続中のディスプレイ一覧を表示して終了
//   ./glasses_passthrough --display N  # N番のディスプレイを使う(--list の番号)
//   ./glasses_passthrough --window     # 通常ウィンドウでSBS合成を確認(グラス未接続でも動く)
// Keys:  x=左右入替  r=90°回転  m=ミラー  s=SBS/両目同一切替  space=一時停止  q/esc=終了
//
import AppKit
import AVFoundation

// 要・最新ファームウェア (MCU 12.1.00.498_20241115)。古い場合は
// https://ota.xreal.com/ultra-update?version=1 でアップデートしてください。
let W = 640, HIMG = 480, HFULL = 482, META_ROW = 480, CTR_COL = 18
// row480 col59 はカメラ識別ビット: 1 = cam0 (is_right=True), 0 = cam1。到着順は当てにしない。
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

/// カメラ毎の縦縞FPN推定と除去(preview_clean.swift と同じオンライン版)
final class Cleaner {
    var stripe = [Float](repeating: 0, count: OW)
    var haveStripe = false
    private var f = [Float](repeating: 0, count: OW * OH)
    private var hp = [Float](repeating: 0, count: OW * OH)

    func clean(_ img: [UInt8]) -> [UInt8] {
        for i in 0..<OW * OH { f[i] = Float(img[i]) }
        let r = 15
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

func cgGray(_ pixels: [UInt8], _ w: Int, _ h: Int) -> CGImage? {
    guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
    return CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: w,
                   space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: 0),
                   provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
}

// ---- stereo view: 左半分=左目, 右半分=右目 に aspect-fit で描画 -----------------
final class StereoView: NSView {
    var left: CGImage?          // 左目に出す画像
    var right: CGImage?         // 右目に出す画像
    var rotationSteps = 0       // 90°単位の回転 (0..3)
    var mirror = false          // 水平ミラー
    var sbs = true              // true: 左右別画像 / false: 両目に同じ画像(2Dモード確認用)
    var overlay = ""            // 左上のデバッグ表示(--window時のみ)
    var showOverlay = false

    override var isFlipped: Bool { true }

    private func drawEye(_ img: CGImage, into rect: NSRect, ctx: CGContext) {
        // rect 内にアスペクト維持で最大化。回転を考慮して収まる矩形を計算。
        let iw = CGFloat(img.width), ih = CGFloat(img.height)
        let rotated = rotationSteps % 2 == 1
        let ew = rotated ? ih : iw, eh = rotated ? iw : ih
        let scale = min(rect.width / ew, rect.height / eh)
        let dw = ew * scale, dh = eh * scale
        let cx = rect.midX, cy = rect.midY
        ctx.saveGState()
        ctx.interpolationQuality = .none
        ctx.translateBy(x: cx, y: cy)
        if mirror { ctx.scaleBy(x: -1, y: 1) }
        ctx.rotate(by: CGFloat(rotationSteps) * .pi / 2)
        // 回転後の「画像座標系」での描画サイズ。isFlipped=true なので上下は既に対応済み。
        let w = rotated ? dh : dw, h = rotated ? dw : dh
        ctx.draw(img, in: NSRect(x: -w / 2, y: -h / 2, width: w, height: h))
        ctx.restoreGState()
    }

    override func draw(_ dirtyRect: NSRect) {
        NSColor.black.setFill(); bounds.fill()
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        let halfW = bounds.width / 2
        let leftRect = NSRect(x: 0, y: 0, width: halfW, height: bounds.height)
        let rightRect = NSRect(x: halfW, y: 0, width: halfW, height: bounds.height)
        if let l = left { drawEye(l, into: leftRect, ctx: ctx) }
        if let r = sbs ? right : left { drawEye(r, into: rightRect, ctx: ctx) }
        if showOverlay && !overlay.isEmpty {
            let attrs: [NSAttributedString.Key: Any] = [
                .foregroundColor: NSColor.green,
                .font: NSFont.monospacedSystemFont(ofSize: 13, weight: .bold)]
            overlay.draw(at: NSPoint(x: 8, y: 6), withAttributes: attrs)
        }
    }
}

// ---- camera -----------------------------------------------------------------
final class Cam: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    weak var view: StereoView?
    var paused = false
    var swapEyes = false        // false: cam0→左目 / true: cam1→左目

    private let luts = [buildLUT(isRight: true), buildLUT(isRight: false)]  // cam0, cam1
    private let cleaners = [Cleaner(), Cleaner()]
    private var latest: [[UInt8]?] = [nil, nil]
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
        out.setSampleBufferDelegate(self, queue: DispatchQueue(label: "xreal.passthrough"))
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
        let cam = 1 - (Int(mono[META_ROW * W + CAM_COL]) & 1)     // テレメトリのカメラ識別ビット
        mono.withUnsafeBufferPointer { buf in
            latest[cam] = cleaners[cam].clean(descramble(buf.baseAddress!, lut: luts[cam]))
        }

        fpsCount += 1
        let dt = Date().timeIntervalSince(fpsStart)
        if dt >= 1 { fps = Double(fpsCount) / dt; fpsCount = 0; fpsStart = Date() }

        guard let c0 = latest[0], let c1 = latest[1] else { return }   // 両カメラ揃うまで待つ
        let leftCam = swapEyes ? c1 : c0
        let rightCam = swapEyes ? c0 : c1
        guard let lImg = cgGray(leftCam, OW, OH), let rImg = cgGray(rightCam, OW, OH) else { return }
        let lbl = String(format: "L|R ctr=%d %.0f fps  x:swap r:rot m:mirror s:sbs q:quit", counter, fps)
        DispatchQueue.main.async {
            self.view?.left = lImg
            self.view?.right = rImg
            self.view?.overlay = lbl
            self.view?.needsDisplay = true
        }
    }
}

// ---- display selection ------------------------------------------------------
func screenID(_ s: NSScreen) -> CGDirectDisplayID {
    (s.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber)?.uint32Value ?? 0
}

func listScreens() {
    for (i, s) in NSScreen.screens.enumerated() {
        let f = s.frame
        let id = screenID(s)
        let builtin = CGDisplayIsBuiltin(id) != 0
        print(String(format: "  [%d] \"%@\"  %dx%d  builtin=%@", i, s.localizedName,
                     Int(f.width), Int(f.height), builtin ? "yes" : "no"))
    }
}

/// グラスのディスプレイを推定: 名前に XREAL/Air/Ultra を含むもの優先、無ければ内蔵以外の外部を1つ。
func pickGlassesScreen() -> NSScreen? {
    let ext = NSScreen.screens.filter { CGDisplayIsBuiltin(screenID($0)) == 0 }
    if let named = ext.first(where: {
        let n = $0.localizedName.lowercased()
        return n.contains("xreal") || n.contains("air") || n.contains("ultra")
    }) { return named }
    return ext.first
}

// ---- entry ------------------------------------------------------------------
let args = CommandLine.arguments
let app = NSApplication.shared            // NSScreen アクセス前に共有インスタンスを用意

if args.contains("--list") {
    print("Displays:")
    listScreens()
    exit(0)
}

let windowMode = args.contains("--window")
var forcedIndex: Int? = nil
if let i = args.firstIndex(of: "--display"), args.count > i + 1 { forcedIndex = Int(args[i + 1]) }

let cam = Cam()

final class AppDelegate: NSObject, NSApplicationDelegate {
    var win: NSWindow!
    var view: StereoView!
    func applicationDidFinishLaunching(_ n: Notification) {
        let target: NSScreen?
        if let idx = forcedIndex, idx >= 0, idx < NSScreen.screens.count {
            target = NSScreen.screens[idx]
        } else {
            target = pickGlassesScreen()
        }

        if windowMode || target == nil {
            if target == nil && !windowMode {
                FileHandle.standardError.write(Data(
                    "外部ディスプレイ(グラス)が見つかりません。--window でウィンドウ確認できます。\n".utf8))
            }
            // 1920x1080 相当のSBSキャンバスを通常ウィンドウで表示
            let w = 1280, h = 360
            win = NSWindow(contentRect: NSRect(x: 100, y: 100, width: w, height: h),
                           styleMask: [.titled, .closable, .miniaturizable, .resizable],
                           backing: .buffered, defer: false)
            win.title = "XREAL Air 2 Ultra — stereo passthrough (window preview)"
            win.contentAspectRatio = NSSize(width: 16, height: 9)
            view = StereoView(frame: NSRect(x: 0, y: 0, width: w, height: h))
            view.showOverlay = true
        } else {
            let f = target!.frame
            print("Fullscreen on \"\(target!.localizedName)\" \(Int(f.width))x\(Int(f.height))")
            win = NSWindow(contentRect: f, styleMask: [.borderless],
                           backing: .buffered, defer: false, screen: target!)
            win.setFrame(f, display: true)
            win.level = .init(rawValue: Int(CGShieldingWindowLevel()))
            win.isOpaque = true
            win.backgroundColor = .black
            view = StereoView(frame: NSRect(origin: .zero, size: f.size))
        }
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
    case "x": cam.swapEyes.toggle()
    case "r": if let v = cam.view { v.rotationSteps = (v.rotationSteps + 1) % 4 }
    case "m": cam.view?.mirror.toggle()
    case "s": cam.view?.sbs.toggle()
    case " ": cam.paused.toggle()
    case "q", "\u{1b}": NSApp.terminate(nil)
    default: break
    }
    cam.view?.needsDisplay = true
    return e
}
app.run()
