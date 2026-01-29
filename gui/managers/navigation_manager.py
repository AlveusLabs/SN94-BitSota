"""导航管理器 - 负责屏幕切换和导航路由"""

from PySide6.QtCore import QObject, Signal, QTimer
from PySide6.QtWidgets import QStackedWidget


class NavigationManager(QObject):
    """管理应用程序的屏幕导航和标签切换"""

    # 信号
    screen_changed = Signal(str)  # screen_name
    show_coming_soon = Signal(str, str)  # title, message
    tab_changed = Signal(str)  # tab_id

    def __init__(
        self,
        content_stack: QStackedWidget,
        screen_stack: QStackedWidget,
        topbar,
        parent=None
    ):
        """
        初始化导航管理器
        
        Args:
            content_stack: 主内容堆栈（包含启动屏幕和应用容器）
            screen_stack: 屏幕堆栈（包含钱包、挖矿、配置屏幕等）
            topbar: 顶部导航栏
            parent: 父对象
        """
        super().__init__(parent)
        self.content_stack = content_stack
        self.screen_stack = screen_stack
        self.topbar = topbar
        
        # 屏幕引用
        self.wallet_screen = None
        self.mining_screen = None
        self.profile_screen = None

    def set_screens(self, wallet_screen, mining_screen, profile_screen):
        """
        设置屏幕引用
        
        Args:
            wallet_screen: 钱包屏幕
            mining_screen: 挖矿屏幕
            profile_screen: 配置屏幕
        """
        self.wallet_screen = wallet_screen
        self.mining_screen = mining_screen
        self.profile_screen = profile_screen

    def handle_start_click(self):
        """处理启动屏幕的开始按钮点击"""
        # 启动屏幕点击后显示用户指南
        # 实际的用户指南显示由 ModalManager 处理
        pass

    def show_main_app(self):
        """显示主应用界面（从启动屏幕切换到主应用）"""
        self.content_stack.setCurrentIndex(1)
        self.screen_changed.emit("main_app")

    def handle_tab_change(self, tab_id: str):
        """
        处理导航标签切换
        
        Args:
            tab_id: 标签ID（setup_wallet, mining, settings, profile）
        """
        if tab_id == "setup_wallet":
            self.navigate_to_wallet()
        elif tab_id == "mining":
            self.navigate_to_mining()
        elif tab_id == "profile":
            # Profile 功能尚未实现，显示即将推出提示
            self.show_coming_soon.emit(
                "Profile Screen",
                "The Profile screen is coming soon! This screen will show your mining "
                "history, rewards, and balances from both Direct Mining and Pool Mining. "
                "You'll be able to view detailed statistics and claim your rewards."
            )
            # 切换回挖矿标签
            self.topbar.set_active_tab("mining")
        elif tab_id == "settings":
            # Settings 功能可以在这里实现
            pass

    def navigate_to_wallet(self):
        """导航到钱包屏幕"""
        if self.wallet_screen:
            self.screen_stack.setCurrentWidget(self.wallet_screen)
            self.screen_changed.emit("wallet")

    def navigate_to_mining(self):
        """导航到挖矿屏幕"""
        if self.mining_screen:
            self.screen_stack.setCurrentWidget(self.mining_screen)
            self.screen_changed.emit("mining")

    def navigate_to_profile(self):
        """导航到配置屏幕"""
        if self.profile_screen:
            self.screen_stack.setCurrentWidget(self.profile_screen)
            self.screen_changed.emit("profile")

    def handle_wallet_connect(self):
        """处理连接钱包请求"""
        self.topbar.set_active_tab("setup_wallet")
        self.navigate_to_wallet()

    def handle_stack_change(self, index: int):
        """
        处理内容堆栈变化
        
        Args:
            index: 堆栈索引
        """
        # 当堆栈页面改变时，可以执行额外的操作
        # 例如更新覆盖层几何等
        pass

    def auto_navigate_to_mining(self):
        """自动导航到挖矿屏幕（用于钱包自动加载后）"""
        self.show_main_app()
        self.topbar.set_active_tab("mining")
        self.navigate_to_mining()
