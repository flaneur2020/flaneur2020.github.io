import SwiftUI

struct ChatDetailView: View {
    let contact: Contact
    @State private var messageText = ""
    @State private var messages = Message.sampleMessages
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            // Custom Header
            HStack(spacing: 12) {
                Button(action: { dismiss() }) {
                    Image(systemName: "arrow.left")
                        .font(.system(size: 22))
                        .foregroundColor(.primary)
                        .frame(width: 32, height: 32)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text(contact.name)
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundColor(.primary)

                    Text(contact.status.rawValue)
                        .font(.system(size: 13))
                        .foregroundColor(.secondary)
                }

                Spacer()

                Button(action: {}) {
                    Image(systemName: "phone")
                        .font(.system(size: 22))
                        .foregroundColor(.primary)
                        .frame(width: 32, height: 32)
                }
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 12)
            .background(Color.white)
            .overlay(
                Rectangle()
                    .frame(height: 1)
                    .foregroundColor(Color.gray.opacity(0.2)),
                alignment: .bottom
            )

            // Messages
            ScrollView {
                VStack(spacing: 16) {
                    ForEach(messages) { message in
                        MessageBubbleView(message: message, contact: contact)
                    }
                }
                .padding(24)
            }

            // Input Section
            VStack(spacing: 12) {
                Divider()

                HStack(spacing: 12) {
                    Button(action: {}) {
                        Image(systemName: "plus")
                            .font(.system(size: 20))
                            .foregroundColor(Color(red: 0.545, green: 0.361, blue: 0.965))
                    }

                    Text("Type a message...")
                        .font(.system(size: 15))
                        .foregroundColor(.gray)
                        .frame(maxWidth: .infinity, alignment: .leading)

                    Button(action: sendMessage) {
                        Image(systemName: "paperplane")
                            .font(.system(size: 20))
                            .foregroundColor(Color(red: 0.545, green: 0.361, blue: 0.965))
                            .frame(width: 32, height: 32)
                    }
                }
                .padding(.horizontal, 16)
                .frame(height: 52)
                .background(Color.gray.opacity(0.1))
                .cornerRadius(26)
                .padding(.horizontal, 16)
            }
            .padding(.bottom, 16)
        }
        .navigationBarHidden(true)
    }

    private func sendMessage() {
        // Add message sending logic
    }
}

struct MessageBubbleView: View {
    let message: Message
    let contact: Contact

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            if !message.isFromCurrentUser {
                // Avatar for received messages
                ZStack {
                    Circle()
                        .fill(avatarColor)
                        .frame(width: 32, height: 32)

                    Text(contact.initials)
                        .font(.system(size: 12, weight: .bold))
                        .fontDesign(.rounded)
                        .foregroundColor(.white)
                }
            } else {
                Spacer()
            }

            // Message Bubble
            VStack(alignment: message.isFromCurrentUser ? .trailing : .leading, spacing: 4) {
                Text(message.text)
                    .font(.system(size: 15))
                    .foregroundColor(message.isFromCurrentUser ? .white : .primary)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(message.isFromCurrentUser ?
                                Color(red: 0.545, green: 0.361, blue: 0.965) :
                                    Color.gray.opacity(0.1))
                    .cornerRadius(20)

                Text(message.formattedTime)
                    .font(.system(size: 12))
                    .foregroundColor(message.isFromCurrentUser ?
                                     Color.white.opacity(0.7) : .gray)
                    .padding(.horizontal, 4)
            }
            .frame(maxWidth: 280, alignment: message.isFromCurrentUser ? .trailing : .leading)

            if message.isFromCurrentUser {
                Spacer()
            }
        }
    }

    private var avatarColor: Color {
        switch contact.avatarColor {
        case "purple": return Color(red: 0.545, green: 0.361, blue: 0.965)
        case "teal": return Color(red: 0.078, green: 0.722, blue: 0.651)
        case "pink": return Color(red: 0.957, green: 0.447, blue: 0.714)
        default: return .blue
        }
    }
}

#Preview {
    ChatDetailView(contact: Contact.sampleContacts[0])
}
