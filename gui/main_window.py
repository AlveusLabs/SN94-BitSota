from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QStackedWidget,
)

from gui.theme import BitSOTATheme
from gui.screens import StartScreen, WalletScreen, MiningScreen, ProfileScreen
from gui.components import TopBar, ModalOverlay
from gui.managers import (
    WalletManager,
    ClientManager,
    NavigationManager,
    ModalManager,
    UpdateManager,
    WindowStyleManager,
)

class MiningWindow(QMainWindow):
    """主窗口 - 协调各个管理器和屏幕"""

    def __init__(self):
        super().__init__()
        
        # 初始化窗口样式管理器
        self.style_manager = WindowStyleManager(self)
        self.style_manager.setup_window()
        
        # 创建 UI
        self._create_ui()
        
        # 应用主题
        self._apply_theme()
        
        # 初始化管理器
        self._initialize_managers()
        
        # 连接信号
        self._connect_signals()
        
        # 尝试自动加载钱包
        self.wallet_manager.auto_load_wallet()
        
        # 设置更新检查
        self.update_manager.setup()

    def showEvent(self, event):
        """窗口显示事件 - 用于设置标题栏颜色"""
        super().showEvent(event)
        self.style_manager.handle_show_event(event)

    def _create_ui(self):
        """创建 UI 组件"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 内容堆栈（启动屏幕 + 主应用）
        self.content_stack = QStackedWidget()
        main_layout.addWidget(self.content_stack)

        # 启动屏幕
        self.start_screen = StartScreen()
        self.content_stack.addWidget(self.start_screen)

        # 主应用容器
        self.app_container = QWidget()
        self.app_container.setObjectName("app_container")
        app_layout = QVBoxLayout(self.app_container)
        app_layout.setContentsMargins(0, 0, 0, 0)
        app_layout.setSpacing(0)

        # 顶部导航栏
        self.topbar = TopBar()
        app_layout.addWidget(self.topbar)

        # 内容区域包装器（带内边距）
        self.content_wrapper = QWidget()
        self.content_wrapper.setObjectName("content_wrapper")
        self.content_wrapper.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        content_wrapper_layout = QVBoxLayout(self.content_wrapper)
        content_wrapper_layout.setContentsMargins(24, 24, 24, 24)

        # 屏幕堆栈
        self.screen_stack = QStackedWidget()
        self.screen_stack.setObjectName("screen_stack")
        self.screen_stack.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        
        self.wallet_screen = WalletScreen()
        self.mining_screen = MiningScreen(main_window=self)
        self.profile_screen = ProfileScreen()
        
        self.screen_stack.addWidget(self.wallet_screen)
        self.screen_stack.addWidget(self.mining_screen)
        self.screen_stack.addWidget(self.profile_screen)
        
        content_wrapper_layout.addWidget(self.screen_stack)
        app_layout.addWidget(self.content_wrapper, 1)

        self.content_stack.addWidget(self.app_container)

        # 创建模态覆盖层
        self.modal_overlay = ModalOverlay(central_widget)
        self.modal_overlay.hide()

        # 保留旧的 sidebar 引用以保持兼容性（但不添加到布局）
        self.sidebar = None

    def _apply_theme(self):
        """应用主题样式"""
        self.setStyleSheet(BitSOTATheme.get_main_stylesheet())
        fonts = BitSOTATheme.get_font_system()
        self.setFont(fonts["primary"])

    def _initialize_managers(self):
        """初始化所有管理器"""
        # 钱包管理器
        self.wallet_manager = WalletManager(self)
        
        # 客户端管理器
        self.client_manager = ClientManager(self)
        
        # 导航管理器
        self.navigation_manager = NavigationManager(
            self.content_stack,
            self.screen_stack,
            self.topbar,
            self
        )
        self.navigation_manager.set_screens(
            self.wallet_screen,
            self.mining_screen,
            self.profile_screen
        )
        
        # 模态框管理器
        self.modal_manager = ModalManager(
            self,
            self.modal_overlay,
            self.content_stack,
            self.screen_stack,
            self.app_container,
            self
        )
        
        # 更新管理器
        self.update_manager = UpdateManager(self, self)

    def _connect_signals(self):
        """连接信号和槽"""
        # 启动屏幕
        self.start_screen.start_clicked.connect(self._on_start_clicked)
        
        # 顶部导航栏
        self.topbar.tab_changed.connect(self.navigation_manager.handle_tab_change)
        self.topbar.user_guide_clicked.connect(self.modal_manager.show_user_guide)
        self.topbar.wallet_clicked.connect(self._on_wallet_dropdown_clicked)
        
        # 钱包屏幕
        self.wallet_screen.wallet_loaded.connect(self._on_wallet_loaded)
        self.wallet_screen.hotkey_imported.connect(self._on_hotkey_imported)
        
        # 内容堆栈变化
        self.content_stack.currentChanged.connect(self._on_stack_changed)
        
        # 钱包管理器
        self.wallet_manager.wallet_loaded.connect(self._on_wallet_manager_loaded)
        self.wallet_manager.hotkey_imported.connect(self._on_wallet_manager_imported)
        self.wallet_manager.wallet_status_updated.connect(self._update_mining_screen_status)
        
        # 模态框管理器
        self.modal_manager.user_guide_completed.connect(self.navigation_manager.show_main_app)
        self.modal_manager.coldkey_address_submitted.connect(self.wallet_manager.save_coldkey_address)
        
        # 导航管理器
        self.navigation_manager.show_coming_soon.connect(self.modal_manager.show_coming_soon)

    def resizeEvent(self, event):
        """窗口大小调整事件"""
        super().resizeEvent(event)
        self.modal_manager.handle_resize_event()

    def _on_stack_changed(self, index):
        """内容堆栈变化事件"""
        self.modal_manager.handle_stack_change()

    def _on_start_clicked(self):
        """启动按钮点击"""
        self.modal_manager.show_user_guide()

    def _on_wallet_dropdown_clicked(self):
        """钱包下拉菜单点击"""
        # 可以在这里显示钱包详情或切换钱包模态框
        pass

    def _on_wallet_loaded(
        self,
        wallet_name: str,
        hotkey_name: str,
        use_existing_coldkey: bool,
        coldkey_address: str
    ):
        """处理从钱包屏幕加载的钱包"""
        success, error = self.wallet_manager.load_wallet(
            wallet_name,
            hotkey_name,
            use_existing_coldkey,
            coldkey_address
        )
        
        if not success:
            self.modal_manager.show_error("Wallet Load Error", error)
            return
        
        # 初始化客户端
        wallet = self.wallet_manager.get_wallet()
        if wallet:
            self.client_manager.initialize_client(wallet)
        
        # 检查是否需要提示输入 coldkey
        if self.wallet_manager.needs_coldkey_prompt(use_existing_coldkey, coldkey_address):
            self.modal_manager.show_coldkey_prompt()

    def _on_hotkey_imported(self, hotkey_name: str, mnemonic: str, coldkey_address: str):
        """处理从钱包屏幕导入的热钥"""
        success, error = self.wallet_manager.import_hotkey(
            hotkey_name,
            mnemonic,
            coldkey_address
        )
        
        if not success:
            self.modal_manager.show_error("Import Failed", error)
            return
        
        # 初始化客户端
        wallet = self.wallet_manager.get_wallet()
        if wallet:
            self.client_manager.initialize_client(wallet)
        
        # 检查是否需要提示输入 coldkey
        if not coldkey_address:
            self.modal_manager.show_coldkey_prompt()

    def _on_wallet_manager_loaded(self, wallet, wallet_name: str, display_address: str):
        """钱包管理器加载钱包后的处理"""
        # 更新顶部栏
        self.topbar.set_wallet_info(wallet_name, display_address)
        
        # 初始化客户端
        if wallet:
            self.client_manager.initialize_client(wallet)
        
        # 如果是自动加载，导航到挖矿屏幕
        # 通过检查当前是否在启动屏幕来判断
        if self.content_stack.currentIndex() == 0:
            self.navigation_manager.auto_navigate_to_mining()

    def _on_wallet_manager_imported(self, wallet, wallet_name: str, display_address: str):
        """钱包管理器导入热钥后的处理"""
        # 更新顶部栏
        self.topbar.set_wallet_info(wallet_name, display_address)
        
        # 初始化客户端
        if wallet:
            self.client_manager.initialize_client(wallet)

    def _update_mining_screen_status(self, wallet_name: str):
        """更新挖矿屏幕状态"""
        if hasattr(self.mining_screen, 'update_wallet_status'):
            self.mining_screen.update_wallet_status(wallet_name)
            self.mining_screen.update_global_sota()

    def get_current_sota(self):
        """获取当前 SOTA 阈值（保留用于向后兼容）"""
        return self.client_manager.fetch_current_sota()

    def _get_relay_endpoint_from_config(self):
        """获取 relay endpoint（保留用于向后兼容）"""
        return self.client_manager.get_relay_endpoint()

    def _prompt_for_coldkey_address(self):
        """提示输入 coldkey 地址（保留用于向后兼容）"""
        self.modal_manager.show_coldkey_prompt()

    # 保留这些属性以保持向后兼容性
    @property
    def wallet(self):
        """获取当前钱包"""
        return self.wallet_manager.get_wallet()

    @property
    def client(self):
        """获取当前客户端"""
        return self.client_manager.get_client()

    @property
    def coldkey_address(self):
        """获取 coldkey 地址"""
        return self.wallet_manager.get_coldkey_address()
    
    def show_modal_with_overlay(self, modal):
        """显示带覆盖层的模态框（保留用于向后兼容）"""
        return self.modal_manager.show_modal(modal)
