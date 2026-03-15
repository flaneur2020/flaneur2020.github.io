import Foundation

// MARK: - Message Model
struct Message: Identifiable {
    let id = UUID()
    let text: String
    let timestamp: Date
    let isFromCurrentUser: Bool

    var formattedTime: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "h:mm a"
        return formatter.string(from: timestamp)
    }
}

// MARK: - Contact Model
struct Contact: Identifiable {
    let id = UUID()
    let name: String
    let initials: String
    let avatarColor: String
    let status: ContactStatus
    let lastMessage: String?
    let lastMessageTime: String?

    enum ContactStatus: String {
        case available = "Available"
        case away = "Away"
        case activeNow = "Active now"

        var color: String {
            switch self {
            case .available, .activeNow:
                return "teal"
            case .away:
                return "gray"
            }
        }
    }
}

// MARK: - Chat Model
struct Chat: Identifiable {
    let id = UUID()
    let contact: Contact
    let messages: [Message]
    let unreadCount: Int
}

// MARK: - Sample Data
extension Contact {
    static let sampleContacts = [
        Contact(
            name: "John Doe",
            initials: "JD",
            avatarColor: "purple",
            status: .available,
            lastMessage: "Hey! Are we still on for today?",
            lastMessageTime: "2m"
        ),
        Contact(
            name: "Sarah Kim",
            initials: "SK",
            avatarColor: "teal",
            status: .away,
            lastMessage: "Sounds good! See you then 👋",
            lastMessageTime: "1h"
        ),
        Contact(
            name: "Emma Wilson",
            initials: "EM",
            avatarColor: "pink",
            status: .available,
            lastMessage: "Did you get my last message?",
            lastMessageTime: "5h"
        )
    ]
}

extension Message {
    static let sampleMessages = [
        Message(
            text: "Hey! How's it going?",
            timestamp: Date().addingTimeInterval(-120),
            isFromCurrentUser: false
        ),
        Message(
            text: "Pretty good! Just finished a project",
            timestamp: Date().addingTimeInterval(-60),
            isFromCurrentUser: true
        ),
        Message(
            text: "We should celebrate tonight! 🎉",
            timestamp: Date().addingTimeInterval(-30),
            isFromCurrentUser: false
        )
    ]
}
