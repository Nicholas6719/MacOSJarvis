import AppKit
import SwiftUI
import WebKit
import AVFoundation

// MARK: - Animated menubar waveform icon

final class WaveformIconView: NSView {

    private let barColor = NSColor(red: 0.0, green: 0.706, blue: 1.0, alpha: 1.0)
    private let barWidth: CGFloat  = 3
    private let barGap: CGFloat    = 3
    private let idleHeights: [CGFloat] = [6, 10, 6]

    private var barHeights: [CGFloat] = [6, 10, 6]
    private var phase: Double = 0
    private var animTimer: Timer?

    override var isFlipped: Bool { false }

    override func draw(_ dirtyRect: NSRect) {
        barColor.setFill()
        let totalWidth = 3 * barWidth + 2 * barGap   // 15
        let startX = (bounds.width - totalWidth) / 2

        for i in 0..<3 {
            let h = barHeights[i]
            let x = startX + CGFloat(i) * (barWidth + barGap)
            let y = (bounds.height - h) / 2
            let rect = NSRect(x: x, y: y, width: barWidth, height: h)
            NSBezierPath(roundedRect: rect, xRadius: 1.5, yRadius: 1.5).fill()
        }
    }

    func startAnimating() {
        guard animTimer == nil else { return }
        animTimer = Timer.scheduledTimer(withTimeInterval: 0.08, repeats: true) { [weak self] _ in
            guard let self else { return }
            self.phase += 0.15
            self.barHeights = [
                7 + 6 * sin(self.phase),
                7 + 6 * sin(self.phase + 1.0),
                7 + 6 * sin(self.phase + 2.0),
            ]
            self.needsDisplay = true
        }
    }

    func stopAnimating() {
        animTimer?.invalidate()
        animTimer = nil
        barHeights = idleHeights
        phase = 0
        needsDisplay = true
    }
}

// MARK: - WKWebView wrapper for the orb

struct WKWebViewRepresentable: NSViewRepresentable {

    func makeNSView(context: Context) -> JarvisWebView {
        let config = WKWebViewConfiguration()
        config.preferences.setValue(true, forKey: "allowFileAccessFromFileURLs")

        let webView = JarvisWebView(frame: .zero, configuration: config)
        webView.navigationDelegate = context.coordinator
        webView.allowsMagnification = false

        if let url = URL(string: "http://localhost:3000") {
            webView.load(URLRequest(url: url))
        }
        return webView
    }

    func updateNSView(_ nsView: JarvisWebView, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator: NSObject, WKNavigationDelegate {
        func webView(_ webView: WKWebView,
                     decidePolicyFor action: WKNavigationAction,
                     decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
            let host = action.request.url?.host ?? ""
            if host == "localhost" || host == "127.0.0.1" || action.navigationType == .other {
                decisionHandler(.allow)
            } else {
                decisionHandler(.cancel)
            }
        }
    }
}

/// Custom WKWebView: allows dragging the window by its background, suppresses context menu.
final class JarvisWebView: WKWebView {
    override var mouseDownCanMoveWindow: Bool { true }
    override func willOpenMenu(_ menu: NSMenu, with event: NSEvent) {
        menu.removeAllItems()
    }
}

// MARK: - Orb content (SwiftUI view that observes BackendManager)

struct OrbContentView: View {
    @EnvironmentObject var backend: BackendManager

    var body: some View {
        ZStack {
            Color.black
            WKWebViewRepresentable()
            VStack {
                Spacer()
                HStack {
                    Spacer()
                    STTOverlay(text: backend.lastSTT)
                        .padding(.trailing, 14)
                        .padding(.bottom, 14)
                }
            }
        }
        .frame(width: 420, height: 420)
    }
}

// MARK: - AppDelegate

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {

    let backendManager = BackendManager()

    private var statusItem: NSStatusItem?
    private var orbWindow: NSPanel?
    private var logsWindow: NSWindow?
    private var waveformView = WaveformIconView(frame: NSRect(x: 0, y: 0, width: 28, height: 18))
    private var isMuted = false
    private var phaseTimer: Timer?

    // MARK: - Lifecycle

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        setupMenubar()
        setupOrbWindow()
        backendManager.start()
        startPhaseObserver()
        requestMicrophonePermission()
    }

    func applicationWillTerminate(_ notification: Notification) {
        phaseTimer?.invalidate()
        backendManager.stop()
    }

    // MARK: - Menubar

    private func setupMenubar() {
        statusItem = NSStatusBar.system.statusItem(withLength: 28)

        if let button = statusItem?.button {
            waveformView.frame = NSRect(
                x: (button.bounds.width - 28) / 2,
                y: (button.bounds.height - 18) / 2,
                width: 28,
                height: 18
            )
            button.addSubview(waveformView)
        }

        let menu = NSMenu()

        let titleItem = NSMenuItem(title: "Jarvis", action: nil, keyEquivalent: "")
        titleItem.attributedTitle = NSAttributedString(
            string: "Jarvis",
            attributes: [.font: NSFont.boldSystemFont(ofSize: 13)]
        )
        titleItem.isEnabled = false
        menu.addItem(titleItem)

        menu.addItem(.separator())

        let statusItem = NSMenuItem(title: "● Idle", action: nil, keyEquivalent: "")
        statusItem.tag = 100
        statusItem.isEnabled = false
        menu.addItem(statusItem)

        menu.addItem(.separator())

        let muteItem = NSMenuItem(title: "Mute", action: #selector(toggleMute), keyEquivalent: "")
        muteItem.tag = 200
        muteItem.target = self
        menu.addItem(muteItem)

        let restartItem = NSMenuItem(title: "Restart Jarvis", action: #selector(restartJarvis), keyEquivalent: "")
        restartItem.target = self
        menu.addItem(restartItem)

        let logsItem = NSMenuItem(title: "Show Logs", action: #selector(showLogs), keyEquivalent: "")
        logsItem.target = self
        menu.addItem(logsItem)

        menu.addItem(.separator())

        let quitItem = NSMenuItem(title: "Quit Jarvis", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        menu.addItem(quitItem)

        self.statusItem?.menu = menu
    }

    // MARK: - Orb window

    private func setupOrbWindow() {
        let panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 420, height: 420),
            styleMask: .borderless,
            backing: .buffered,
            defer: false
        )
        panel.level = .floating
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.isMovableByWindowBackground = true
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]

        // Position bottom-right of screen
        if let screen = NSScreen.main?.visibleFrame {
            panel.setFrameOrigin(NSPoint(x: screen.maxX - 440, y: screen.minY + 20))
        }

        let orbContent = OrbContentView()
            .environmentObject(backendManager)

        panel.contentView = NSHostingView(rootView: orbContent)
        orbWindow = panel
        panel.orderFrontRegardless()
    }

    // MARK: - Phase observer

    private func startPhaseObserver() {
        phaseTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                let menu = self.statusItem?.menu
                let statusMenuItem = menu?.item(withTag: 100)

                switch self.backendManager.phase {
                case .idle:
                    self.waveformView.stopAnimating()
                    statusMenuItem?.title = "● Idle"
                case .starting:
                    self.waveformView.stopAnimating()
                    statusMenuItem?.title = "● Starting…"
                case .ready:
                    self.waveformView.startAnimating()
                    statusMenuItem?.title = "● Ready"
                case .failed:
                    self.waveformView.stopAnimating()
                    statusMenuItem?.title = "● Error"
                }
            }
        }
    }

    // MARK: - Menu actions

    @objc private func toggleMute() {
        Task { @MainActor in
            self.isMuted.toggle()
            if let item = self.statusItem?.menu?.item(withTag: 200) {
                item.title = self.isMuted ? "Unmute" : "Mute"
            }
            guard let url = URL(string: "http://localhost:3000/api/mute") else { return }
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try? JSONSerialization.data(withJSONObject: ["muted": self.isMuted])
            _ = try? await URLSession.shared.data(for: request)
        }
    }

    @objc private func restartJarvis() {
        Task { @MainActor in
            self.backendManager.restart()
        }
    }

    @objc private func showLogs() {
        if logsWindow == nil {
            let logsView = LogsView().environmentObject(backendManager)
            let hosting = NSHostingView(rootView: logsView)
            hosting.frame = NSRect(x: 0, y: 0, width: 700, height: 400)

            let window = NSWindow(
                contentRect: NSRect(x: 0, y: 0, width: 700, height: 400),
                styleMask: [.titled, .closable, .resizable, .miniaturizable],
                backing: .buffered,
                defer: false
            )
            window.title = "Jarvis — Logs"
            window.contentView = hosting
            window.center()
            window.isReleasedWhenClosed = false
            logsWindow = window
        }
        logsWindow?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    // MARK: - Microphone

    private func requestMicrophonePermission() {
        AVCaptureDevice.requestAccess(for: .audio) { _ in }
    }
}
