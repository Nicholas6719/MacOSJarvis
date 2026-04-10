import SwiftUI

struct ContentView: View {
    var body: some View {
        EmptyView()
    }
}

struct STTOverlay: View {
    let text: String
    var body: some View {
        if !text.isEmpty {
            Text(text)
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundColor(.white.opacity(0.9))
                .lineLimit(3)
                .multilineTextAlignment(.leading)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .frame(maxWidth: 280, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(.ultraThinMaterial.opacity(0.85))
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.white.opacity(0.12), lineWidth: 1)
                        )
                )
                .shadow(color: .black.opacity(0.4), radius: 6, x: 0, y: 2)
                .transition(.opacity.combined(with: .scale(scale: 0.96,
                    anchor: .bottomTrailing)))
                .animation(.easeInOut(duration: 0.25), value: text)
        }
    }
}
