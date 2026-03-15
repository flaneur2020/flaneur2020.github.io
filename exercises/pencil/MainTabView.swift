import SwiftUI

struct MainTabView: View {
    @State private var selectedTab = 0

    var body: some View {
        ZStack(alignment: .bottom) {
            // Tab Content
            TabView(selection: $selectedTab) {
                ChatListView()
                    .tag(0)

                ContactsView()
                    .tag(1)

                SettingsView()
                    .tag(2)
            }
            .tabViewStyle(.page(indexDisplayMode: .never))

            // Custom Tab Bar with Gradient
            VStack(spacing: 0) {
                // Gradient Fade
                LinearGradient(
                    gradient: Gradient(colors: [
                        Color.white.opacity(0),
                        Color.white
                    ]),
                    startPoint: .top,
                    endPoint: .bottom
                )
                .frame(height: 30)

                // Pill-shaped Tab Bar
                HStack(spacing: 0) {
                    TabBarButton(
                        icon: "message",
                        title: "Chat",
                        isSelected: selectedTab == 0,
                        action: { selectedTab = 0 }
                    )

                    TabBarButton(
                        icon: "person.2",
                        title: "Contacts",
                        isSelected: selectedTab == 1,
                        action: { selectedTab = 1 }
                    )

                    TabBarButton(
                        icon: "gearshape",
                        title: "Settings",
                        isSelected: selectedTab == 2,
                        action: { selectedTab = 2 }
                    )
                }
                .frame(height: 62)
                .padding(4)
                .background(Color.gray.opacity(0.1))
                .cornerRadius(100)
                .padding(.horizontal, 21)
                .padding(.bottom, 21)
            }
            .background(Color.white.opacity(0.01))
        }
        .ignoresSafeArea(.keyboard)
    }
}

struct TabBarButton: View {
    let icon: String
    let title: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 24))
                    .foregroundColor(isSelected ?
                                     Color(red: 0.545, green: 0.361, blue: 0.965) :
                                        Color.gray)

                Text(title)
                    .font(.system(size: 11, weight: isSelected ? .semibold : .medium))
                    .foregroundColor(isSelected ?
                                     Color(red: 0.545, green: 0.361, blue: 0.965) :
                                        Color.gray)
            }
            .frame(maxWidth: .infinity)
        }
    }
}

struct SettingsView: View {
    var body: some View {
        VStack {
            Text("Settings")
                .font(.system(size: 34, weight: .bold))
                .fontDesign(.rounded)

            Spacer()
        }
        .padding(.horizontal, 24)
    }
}

#Preview {
    MainTabView()
}
