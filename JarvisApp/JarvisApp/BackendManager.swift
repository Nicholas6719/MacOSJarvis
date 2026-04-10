import Foundation
import Darwin

@MainActor
final class BackendManager: ObservableObject {

    // MARK: - Phase

    enum Phase: Equatable {
        case idle
        case starting   // backend booting, waiting for :3000
        case ready
        case failed(String)

        static func == (lhs: Phase, rhs: Phase) -> Bool {
            switch (lhs, rhs) {
            case (.idle, .idle), (.starting, .starting), (.ready, .ready):
                return true
            case (.failed(let a), .failed(let b)):
                return a == b
            default:
                return false
            }
        }

        var failureMessage: String? {
            if case .failed(let m) = self { return m }
            return nil
        }
    }

    // MARK: - Published

    @Published var phase:   Phase  = .idle
    @Published var logs:    String = ""   // backend runtime logs
    @Published var lastSTT: String = ""   // latest voice transcription

    // MARK: - Private

    private var process:       Process?
    private var stdoutBuffer = ""
    private var sttClearTask: Task<Void, Never>?

    // MARK: - Hardcoded paths (existing working venv)

    private let venvPython = "/Users/nicholascoppola/Documents/Coding_Projects/Jarvis/.venv311/bin/python"
    private let scriptPath = "/Users/nicholascoppola/Documents/Coding_Projects/Jarvis/chatbot_speech_to_speech.py"
    private let workingDir = "/Users/nicholascoppola/Documents/Coding_Projects/Jarvis"

    // MARK: - Entry point

    func start() {
        guard process == nil else { return }
        switch phase {
        case .starting, .ready: return
        default: break
        }

        let fm = FileManager.default
        guard fm.fileExists(atPath: venvPython) else {
            phase = .failed("Python not found at \(venvPython)")
            return
        }
        guard fm.fileExists(atPath: scriptPath) else {
            phase = .failed("Script not found at \(scriptPath)")
            return
        }

        startBackend()
    }

    // MARK: - Backend launch

    private func startBackend() {
        guard process == nil else { return }
        phase = .starting

        let proc = Process()
        proc.executableURL       = URL(fileURLWithPath: venvPython)
        proc.arguments           = [scriptPath]
        proc.currentDirectoryURL = URL(fileURLWithPath: workingDir)

        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        // Prepend venv/bin to PATH so subprocesses find the right Python
        let venvBin = (venvPython as NSString).deletingLastPathComponent
        env["PATH"] = "\(venvBin):\(env["PATH"] ?? "/usr/bin:/bin")"
        proc.environment = env

        let outPipe = Pipe()
        let errPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError  = errPipe

        outPipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            let data = h.availableData
            guard !data.isEmpty, let s = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor [weak self] in self?.processStdout(s) }
        }
        errPipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            let data = h.availableData
            guard !data.isEmpty, let s = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor [weak self] in self?.appendLog(s) }
        }
        proc.terminationHandler = { [weak self] p in
            outPipe.fileHandleForReading.readabilityHandler = nil
            errPipe.fileHandleForReading.readabilityHandler = nil
            let code = p.terminationStatus
            Task { @MainActor [weak self] in
                self?.process = nil
                self?.appendLog("\n[Backend exited — code \(code)]\n")
                if case .ready = self?.phase { self?.phase = .idle }
            }
        }

        do {
            try proc.run()
            process = proc
            appendLog("▶  Backend started (PID \(proc.processIdentifier))\n")
            appendLog("   Working dir: \(workingDir)\n\n")
            pollForReady()
        } catch {
            phase = .failed("Launch failed: \(error.localizedDescription)")
        }
    }

    // MARK: - Polling for readiness

    private func pollForReady() {
        Task { @MainActor in
            for _ in 0 ..< 1_200 {   // up to 10 minutes
                guard process != nil else { return }
                if await checkPort3000() {
                    phase = .ready
                    return
                }
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
            if phase == .starting {
                phase = .failed("Backend didn't respond on :3000 after 10 min.\nCheck Show Logs for details.")
            }
        }
    }

    private func checkPort3000() async -> Bool {
        guard let url = URL(string: "http://127.0.0.1:3000/api/status") else { return false }
        var req = URLRequest(url: url, timeoutInterval: 3)
        req.httpMethod = "GET"
        do {
            let (_, res) = try await URLSession.shared.data(for: req)
            return (res as? HTTPURLResponse)?.statusCode == 200
        } catch { return false }
    }

    // MARK: - Stop / Restart

    func stop() {
        guard let proc = process else { return }
        Darwin.kill(proc.processIdentifier, SIGINT)
        let p = proc
        DispatchQueue.global().asyncAfter(deadline: .now() + 4) {
            if p.isRunning { Darwin.kill(p.processIdentifier, SIGKILL) }
        }
        process = nil
        phase   = .idle
    }

    func restart() {
        stop()
        phase = .idle
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            self.start()
        }
    }

    // MARK: - Log helpers

    func appendLog(_ text: String) {
        logs += text
        if logs.count > 100_000 { logs = String(logs.suffix(80_000)) }
    }
    func clearLogs() { logs = "" }

    private func processStdout(_ text: String) {
        appendLog(text)
        stdoutBuffer += text
        var lines = stdoutBuffer.components(separatedBy: "\n")
        stdoutBuffer = lines.removeLast()
        for line in lines {
            let t = line.trimmingCharacters(in: .whitespaces)
            if t.hasPrefix("You: ") {
                let stt = String(t.dropFirst(5))
                guard !stt.isEmpty else { continue }
                lastSTT = stt
                scheduleSTTClear()
            }
        }
    }

    private func scheduleSTTClear() {
        sttClearTask?.cancel()
        sttClearTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 8_000_000_000)
            if !Task.isCancelled { self.lastSTT = "" }
        }
    }
}
