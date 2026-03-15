import SwiftUI

struct ChatListView: View {
    @State private var searchText = ""
    let contacts = Contact.sampleContacts

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Custom Header
                HStack {
                    Text("Messages")
                        .font(.system(size: 34, weight: .bold))
                        .fontDesign(.rounded)

                    Spacer()

                    Button(action: {}) {
                        Image(systemName: "bell")
                            .font(.system(size: 22))
                            .foregroundColor(.primary)
                            .frame(width: 48, height: 48)
                            .background(Color.gray.opacity(0.1))
                            .clipShape(Circle())
                    }
                }
                .padding(.horizontal, 24)
                .padding(.top, 0)

                // Search Bar
                HStack(spacing: 14) {
                    Image(systemName: "magnifyingglass")
                        .foregroundColor(.gray)
                        .font(.system(size: 20))

                    Text("Search messages...")
                        .foregroundColor(.gray)
                        .font(.system(size: 15))

                    Spacer()
                }
                .frame(height: 52)
                .padding(.horizontal, 20)
                .background(Color.gray.opacity(0.1))
                .cornerRadius(26)
                .padding(.horizontal, 24)
                .padding(.top, 32)

                // Chat List
                VStack(spacing: 0) {
                    ForEach(contacts) { contact in
                        NavigationLink(destination: ChatDetailView(contact: contact)) {
                            ChatRowView(contact: contact)
                        }
                        .buttonStyle(PlainButtonStyle())
                    }
                }
                .background(Color.gray.opacity(0.1))
                .cornerRadius(24)
                .padding(.horizontal, 24)
                .padding(.top, 32)

                Spacer()
            }
            .navigationBarHidden(true)
        }
    }
}

struct ChatRowView: View {
    let contact: Contact

    var body: some View {
        HStack(spacing: 14) {
            // Avatar
            ZStack {
                Circle()
                    .fill(avatarColor)
                    .frame(width: 56, height: 56)

                Text(contact.initials)
                    .font(.system(size: 18, weight: .bold))
                    .fontDesign(.rounded)
                    .foregroundColor(.white)
            }

            // Content
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(contact.name)
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundColor(.primary)

                    Spacer()

                    Text(contact.lastMessageTime ?? "")
                        .font(.system(size: 13))
                        .foregroundColor(.secondary)
                }

                Text(contact.lastMessage ?? "")
                    .font(.system(size: 14))
                    .foregroundColor(.secondary)
                    .lineLimit(1)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 18)
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
    ChatListView()
}
