# iOS Chat App - SwiftUI Implementation

这是一个iOS原生风格的聊天应用，包含多个页面和现代化的UI设计。

## 📱 设计预览

设计文件位于 `codingapp.pen`，包含三个主要页面：
1. **聊天列表页** - 显示所有对话
2. **聊天详情页** - 单个对话界面
3. **联系人页** - 联系人列表

## 🏗️ 项目结构

```
.
├── ChatApp.swift           # 应用入口
├── MainTabView.swift       # 主标签导航
├── Models.swift            # 数据模型
├── ChatListView.swift      # 聊天列表视图
├── ChatDetailView.swift    # 聊天详情视图
├── ContactsView.swift      # 联系人视图
└── codingapp.pen          # 设计文件
```

## ✨ 主要特性

### 1. 现代化设计
- iOS原生风格的圆角设计 (24px radius)
- 紫色主题色 (#8B5CF6)
- 柔和的灰色背景 (#F4F4F5)
- 渐变式底部导航栏

### 2. 三个核心页面

#### 聊天列表页 (ChatListView)
- 顶部通知按钮
- 搜索栏
- 聊天预览卡片
- 显示最后消息和时间
- 彩色头像圆圈

#### 聊天详情页 (ChatDetailView)
- 返回按钮
- 联系人状态显示
- 消息气泡（左右对齐）
- 时间戳
- 底部输入框
- 附件和发送按钮

#### 联系人页 (ContactsView)
- 添加联系人按钮
- 搜索功能
- 联系人状态显示 (Available/Away)
- 快速消息按钮

### 3. 自定义底部导航
- 药丸形状设计 (100px radius)
- 渐变过渡效果
- 三个标签：聊天、联系人、设置
- 选中状态紫色高亮

## 🎨 设计规范

### 颜色系统
```swift
主色调 (紫色): #8B5CF6
青色 (在线):   #14B8A6
粉色 (强调):   #F472B6
卡片背景:      #F4F4F5
文字主色:      #18181B
文字次色:      #71717A
文字提示:      #A1A1AA
```

### 字体规范
- **标题**: Plus Jakarta Sans (34px, 粗体)
- **正文**: Inter (15-16px)
- **标签**: Inter (11-13px)

### 圆角系统
- 卡片/按钮: 24px
- 搜索栏: 26px
- 消息气泡: 20px
- 底部导航: 100px (药丸形)
- 头像: 完全圆形

## 🚀 如何使用

### 在Xcode中运行

1. 创建新的Xcode项目 (iOS App, SwiftUI)
2. 复制所有 `.swift` 文件到项目中
3. 确保 `ChatApp.swift` 包含 `@main` 标记
4. 运行项目 (⌘R)

### 文件说明

#### Models.swift
定义了核心数据模型：
- `Message`: 消息模型
- `Contact`: 联系人模型
- 示例数据

#### ChatListView.swift
聊天列表页面，包括：
- 自定义顶部导航
- 搜索栏
- 聊天列表行

#### ChatDetailView.swift
对话详情页面，包括：
- 自定义导航栏
- 消息气泡组件
- 输入框

#### ContactsView.swift
联系人列表页面，包括：
- 联系人行组件
- 状态指示器

#### MainTabView.swift
主导航控制器，包括：
- 自定义药丸形标签栏
- 渐变效果
- 标签切换逻辑

## 📝 代码亮点

### 1. 自定义导航栏
所有页面使用 `.navigationBarHidden(true)` 隐藏默认导航栏，实现完全自定义的UI。

### 2. 消息气泡对齐
```swift
HStack {
    if !message.isFromCurrentUser {
        Avatar()
    } else {
        Spacer()
    }
    MessageBubble()
}
```

### 3. 药丸形标签栏
```swift
.cornerRadius(100)  // 完整的药丸形状
```

### 4. 渐变过渡
```swift
LinearGradient(
    gradient: Gradient(colors: [
        Color.white.opacity(0),
        Color.white
    ]),
    startPoint: .top,
    endPoint: .bottom
)
```

## 🔧 自定义建议

### 添加更多功能
1. 实际的消息发送逻辑
2. 数据持久化 (CoreData/SwiftData)
3. 推送通知
4. 图片/视频消息
5. 语音消息
6. 用户认证

### UI增强
1. 滑动删除聊天
2. 长按消息菜单
3. 打字指示器
4. 已读状态
5. 消息搜索
6. 主题切换

## 📐 设计一致性

所有UI元素严格遵循iOS设计规范：
- ✅ 圆角一致性
- ✅ 间距系统化 (16px, 24px, 32px)
- ✅ 颜色语义化
- ✅ 字体层级清晰
- ✅ 触摸区域合理 (最小44x44pt)

## 🎯 设计决策

### 为什么选择这些颜色？
- **紫色** (#8B5CF6): 现代、友好、充满活力
- **青色** (#14B8A6): 表示在线/可用状态
- **粉色** (#F472B6): 作为强调色增加趣味性

### 为什么使用药丸形标签栏？
- 更现代的视觉效果
- 与整体圆润设计一致
- 提供更好的视觉层次

### 为什么自定义导航栏？
- 完全控制布局和间距
- 与设计稿完美匹配
- 更大的标题字体增强视觉冲击力

## 📱 适配说明

当前设计针对 iPhone 标准尺寸 (402px 宽)，可以轻松适配：
- iPhone 14/15 Pro Max
- iPhone 14/15 Pro
- iPhone 14/15
- iPhone SE

建议使用 SwiftUI 的自适应布局特性确保在不同设备上都能良好显示。

---

**设计 & 代码**: 使用 Pencil 设计工具和 SwiftUI 实现
**风格指南**: iOS 原生设计规范
**最后更新**: 2026-01-29
