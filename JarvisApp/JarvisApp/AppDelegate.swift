import AppKit
import SwiftUI
import WebKit
import AVFoundation

// MARK: - Animated menubar waveform icon

final class WaveformIconView: NSView {

    private let blueColor = NSColor(red: 0.0, green: 0.706, blue: 1.0, alpha: 1.0)
    private let grayColor = NSColor.gray.withAlphaComponent(0.6)
    private let barWidth: CGFloat  = 3
    private let barGap: CGFloat    = 3
    private let idleHeights: [CGFloat] = [6, 10, 6]

    private var barHeights: [CGFloat] = [6, 10, 6]
    private var phase: Double = 0
    private var animTimer: Timer?
    var isPaused: Bool = false

    override var isFlipped: Bool { false }

    override func draw(_ dirtyRect: NSRect) {
        let color = isPaused ? grayColor : blueColor
        color.setFill()
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

    func setPaused(_ paused: Bool) {
        isPaused = paused
        if paused {
            stopAnimating()
        }
        needsDisplay = true
    }
}

// MARK: - WKWebView wrapper for the orb

struct WKWebViewRepresentable: NSViewRepresentable {

    /// Callback so AppDelegate can store a reference to the WKWebView for forced reloads.
    var onWebViewCreated: ((JarvisWebView) -> Void)?

    func makeNSView(context: Context) -> JarvisWebView {
        let config = WKWebViewConfiguration()
        config.preferences.setValue(true, forKey: "allowFileAccessFromFileURLs")

        let webView = JarvisWebView(frame: .zero, configuration: config)
        webView.navigationDelegate = context.coordinator
        webView.allowsMagnification = false

        if let url = URL(string: "http://localhost:3000") {
            webView.load(URLRequest(url: url))
        }

        onWebViewCreated?(webView)
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

        /// Retry on connection refused (backend not ready yet).
        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                if let url = URL(string: "http://localhost:3000") {
                    webView.load(URLRequest(url: url))
                }
            }
        }

        /// Retry on mid-load failure.
        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                if let url = URL(string: "http://localhost:3000") {
                    webView.load(URLRequest(url: url))
                }
            }
        }
    }
}

/// Custom WKWebView: allows dragging the window by its background, suppresses context menu.
final class JarvisWebView: WKWebView {
    override var mouseDownCanMoveWindow: Bool { true }
    override func willOpenMenu(_ menu: NSMenu, with event: NSEvent) {
        // Don't suppress if this is the orb resize context menu (has items with tags/targets)
        if menu.items.contains(where: { $0.target != nil }) { return }
        menu.removeAllItems()
    }
}

// MARK: - Orb content (SwiftUI view that observes BackendManager)

struct OrbContentView: View {
    @EnvironmentObject var backend: BackendManager
    var onWebViewCreated: ((JarvisWebView) -> Void)?

    var body: some View {
        ZStack {
            Color.black
            WKWebViewRepresentable(onWebViewCreated: onWebViewCreated)
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

// MARK: - Menu state enum

enum JarvisMenuState {
    case starting
    case ready
    case paused
    case error
}

// MARK: - AppDelegate

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {

    let backendManager = BackendManager()

    private var statusItem: NSStatusItem?
    private var orbWindow: NSPanel?
    private var orbWebView: JarvisWebView?
    private var logsWindow: NSWindow?
    private var waveformView = WaveformIconView(frame: NSRect(x: 0, y: 0, width: 28, height: 18))
    private var isMuted = false
    private var isPaused = false
    private var phaseTimer: Timer?
    private var lastObservedPhase: BackendManager.Phase = .idle

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

    func applicationDidBecomeActive(_ notification: Notification) {
        if !isPaused { orbWindow?.orderFrontRegardless() }
    }

    func applicationWillResignActive(_ notification: Notification) {
        if !isPaused { orbWindow?.orderFrontRegardless() }
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

        // 1. Header
        let titleItem = NSMenuItem(title: "Jarvis", action: nil, keyEquivalent: "")
        titleItem.attributedTitle = NSAttributedString(
            string: "Jarvis",
            attributes: [.font: NSFont.boldSystemFont(ofSize: 13)]
        )
        titleItem.isEnabled = false
        menu.addItem(titleItem)

        // 2. Separator
        menu.addItem(.separator())

        // 3. Status indicator
        let statusMenuItem = NSMenuItem(title: "● Starting…", action: nil, keyEquivalent: "")
        statusMenuItem.tag = 100
        statusMenuItem.isEnabled = false
        menu.addItem(statusMenuItem)

        // 4. Separator
        menu.addItem(.separator())

        // 5. Pause / Resume
        let pauseItem = NSMenuItem(title: "Pause Jarvis", action: #selector(togglePause), keyEquivalent: "")
        pauseItem.tag = 300
        pauseItem.target = self
        pauseItem.isEnabled = false  // disabled while starting
        menu.addItem(pauseItem)

        // 6. Restart
        let restartItem = NSMenuItem(title: "Restart Jarvis", action: #selector(restartJarvis), keyEquivalent: "")
        restartItem.target = self
        menu.addItem(restartItem)

        // 7. Separator
        menu.addItem(.separator())

        // 8. Mute
        let muteItem = NSMenuItem(title: "Mute", action: #selector(toggleMute), keyEquivalent: "")
        muteItem.tag = 200
        muteItem.target = self
        menu.addItem(muteItem)

        // 9. Show Logs
        let logsItem = NSMenuItem(title: "Show Logs", action: #selector(showLogs), keyEquivalent: "")
        logsItem.target = self
        menu.addItem(logsItem)

        // 10. Show Orb
        let showOrbItem = NSMenuItem(title: "Show Orb", action: #selector(showOrb), keyEquivalent: "")
        showOrbItem.tag = 400
        showOrbItem.target = self
        menu.addItem(showOrbItem)

        // 11. Separator
        menu.addItem(.separator())

        // 12. Quit
        let quitItem = NSMenuItem(title: "Quit Jarvis", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        menu.addItem(quitItem)

        self.statusItem?.menu = menu
    }

    // MARK: - Menu state helper

    private func updateMenuForState(_ state: JarvisMenuState) {
        guard let menu = statusItem?.menu else { return }

        if let item = menu.item(withTag: 100) {
            switch state {
            case .starting: item.title = "● Starting…"
            case .ready:    item.title = "● Active"
            case .paused:   item.title = "● Paused"
            case .error:    item.title = "● Error"
            }
        }

        if let pauseItem = menu.item(withTag: 300) {
            switch state {
            case .starting:
                pauseItem.title = "Pause Jarvis"
                pauseItem.isEnabled = false
            case .ready:
                pauseItem.title = "Pause Jarvis"
                pauseItem.isEnabled = true
            case .paused:
                pauseItem.title = "Resume Jarvis"
                pauseItem.isEnabled = true
            case .error:
                pauseItem.title = "Pause Jarvis"
                pauseItem.isEnabled = false
            }
        }
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
        panel.hidesOnDeactivate = false
        panel.canHide = false
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]

        // Position bottom-right of screen
        if let screen = NSScreen.main?.visibleFrame {
            panel.setFrameOrigin(NSPoint(x: screen.maxX - 440, y: screen.minY + 20))
        }

        let orbContent = OrbContentView(onWebViewCreated: { [weak self] webView in
            self?.orbWebView = webView
        })
        .environmentObject(backendManager)

        panel.contentView = NSHostingView(rootView: orbContent)
        panel.contentView?.menu = makeOrbContextMenu()
        orbWindow = panel
        panel.orderFrontRegardless()
    }

    // MARK: - Orb context menu (right-click resize)

    private func makeOrbContextMenu() -> NSMenu {
        let menu = NSMenu()

        let smallItem = NSMenuItem(title: "Small (280×280)", action: #selector(setOrbSmall), keyEquivalent: "")
        smallItem.target = self
        menu.addItem(smallItem)

        let mediumItem = NSMenuItem(title: "Medium (420×420)", action: #selector(setOrbMedium), keyEquivalent: "")
        mediumItem.target = self
        menu.addItem(mediumItem)

        let largeItem = NSMenuItem(title: "Large (560×560)", action: #selector(setOrbLarge), keyEquivalent: "")
        largeItem.target = self
        menu.addItem(largeItem)

        menu.addItem(.separator())

        let hideItem = NSMenuItem(title: "Hide Orb", action: #selector(hideOrb), keyEquivalent: "")
        hideItem.target = self
        menu.addItem(hideItem)

        return menu
    }

    @objc private func setOrbSmall()  { resizeOrb(to: 280) }
    @objc private func setOrbMedium() { resizeOrb(to: 420) }
    @objc private func setOrbLarge()  { resizeOrb(to: 560) }

    @objc private func hideOrb() {
        orbWindow?.orderOut(nil)
    }

    @objc private func showOrb() {
        orbWindow?.orderFrontRegardless()
    }

    private func resizeOrb(to size: CGFloat) {
        guard let panel = orbWindow, let screen = NSScreen.main else { return }
        let frame = NSRect(
            x: screen.visibleFrame.maxX - size - 20,
            y: screen.visibleFrame.minY + 20,
            width: size,
            height: size
        )
        panel.setFrame(frame, display: true, animate: true)
    }

    // MARK: - Phase observer

    private func startPhaseObserver() {
        phaseTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }

                let currentPhase = self.backendManager.phase
                let wasReady = self.lastObservedPhase == .ready
                self.lastObservedPhase = currentPhase

                switch currentPhase {
                case .idle:
                    if !self.isPaused {
                        self.updateMenuForState(.starting)
                    }
                    self.waveformView.stopAnimating()
                case .starting:
                    self.updateMenuForState(.starting)
                    self.waveformView.stopAnimating()
                case .ready:
                    if !self.isPaused {
                        self.updateMenuForState(.ready)
                        self.waveformView.setPaused(false)
                        self.waveformView.startAnimating()
                    }
                    // Force a fresh load once when backend first becomes ready
                    if !wasReady, let url = URL(string: "http://localhost:3000") {
                        self.orbWebView?.load(URLRequest(url: url))
                    }
                case .failed:
                    self.updateMenuForState(.error)
                    self.waveformView.stopAnimating()
                }
            }
        }
    }

    // MARK: - Menu actions

    @objc private func togglePause() {
        Task { @MainActor in
            if self.isPaused {
                // Resume
                self.isPaused = false
                self.orbWindow?.orderFrontRegardless()
                self.backendManager.start()
                self.updateMenuForState(.starting)
            } else {
                // Pause
                self.isPaused = true
                self.backendManager.stop()
                self.orbWindow?.orderOut(nil)
                self.updateMenuForState(.paused)
                self.waveformView.setPaused(true)
            }
        }
    }

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
            self.isPaused = false
            self.waveformView.setPaused(false)
            self.backendManager.restart()
            self.updateMenuForState(.starting)
            // Give backend a head start then trigger a reload
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            if let url = URL(string: "http://localhost:3000") {
                self.orbWebView?.load(URLRequest(url: url))
            }
            self.orbWindow?.orderFrontRegardless()
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
