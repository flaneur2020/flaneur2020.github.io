import SwiftUI

struct ContactsView: View {
    @State private var searchText = ""
    let contacts = Contact.sampleContacts

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Custom Header
                HStack {
                    Text("Contacts")
                        .font(.system(size: 34, weight: .bold))
                        .fontDesign(.rounded)

                    Spacer()

                    Button(action: {}) {
                        Image(systemName: "plus")
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

                    Text("Search contacts...")
                        .foregroundColor(.gray)
                        .font(.system(size: 15))

                    Spacer()
                }
                .frame(height: 52)
                .padding(.horizontal, 20)
                .background(Color.gray.opacity(0.1))
                .cornerRadius(26)
                .padding(.horizontal, 24)
                .padding(.top, 24)

                // Contacts List
                VStack(spacing: 0) {
                    ForEach(contacts) { contact in
                        ContactRowView(contact: contact)
                    }
                }
                .background(Color.gray.opacity(0.1))
                .cornerRadius(24)
                .padding(.horizontal, 24)
                .padding(.top, 24)

                Spacer()
            }
            .navigationBarHidden(true)
        }
    }
}

struct ContactRowView: View {
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

            // Contact Info
            VStack(alignment: .leading, spacing: 4) {
                Text(contact.name)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundColor(.primary)

                Text(contact.status.rawValue)
                    .font(.system(size: 13))
                    .foregroundColor(statusColor)
            }

            Spacer()

            // Action Button
            Button(action: {}) {
                Image(systemName: "message")
                    .font(.system(size: 20))
                    .foregroundColor(Color(red: 0.545, green: 0.361, blue: 0.965))
                    .frame(width: 32, height: 32)
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

    private var statusColor: Color {
        switch contact.status {
        case .available, .activeNow:
            return Color(red: 0.078, green: 0.722, blue: 0.651)
        case .away:
            return .secondary
        }
    }
}

#Preview {
    ContactsView()
}
