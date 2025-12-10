from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QStackedWidget,
    QApplication,
)
from typing import Optional
import webbrowser

from bittensor_network.wallet import Wallet
from miner.client import BittensorDirectClient
from gui.theme import BitSOTATheme
from gui.screens import StartScreen, WalletScreen, MiningScreen, ProfileScreen
from gui.components import Sidebar, UserGuideModal, InviteCodeModal, ColdkeyAddressModal, ComingSoonModal, UpdateAvailableModal
from gui.resource_path import resource_path
from gui.update_checker import UpdateChecker


class MiningWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.wallet: Optional[Wallet] = None
        self.client: Optional[BittensorDirectClient] = None
        self.coldkey_address: Optional[str] = None
        self.update_checker = UpdateChecker()
        self._setup_window()
        self._create_ui()
        self._apply_theme()
        self._try_auto_load_wallet()
        self._setup_update_checker()

    def _setup_window(self):
        self.setWindowTitle("BitSota")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        screen = QApplication.primaryScreen().geometry()
        window_geometry = self.frameGeometry()
        center_point = screen.center()
        window_geometry.moveCenter(center_point)
        self.move(window_geometry.topLeft())

    def _create_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.content_stack = QStackedWidget()
        main_layout.addWidget(self.content_stack)

        self.start_screen = StartScreen()
        self.start_screen.start_clicked.connect(self._on_start_clicked)
        self.content_stack.addWidget(self.start_screen)

        app_container = QWidget()
        app_container.setObjectName("app_container")
        app_layout = QHBoxLayout(app_container)
        app_layout.setContentsMargins(0, 0, 0, 0)
        app_layout.setSpacing(0)

        self.sidebar = Sidebar()
        self.sidebar.add_tab("setup_wallet", "Setup Wallet", resource_path("gui/images/Wallet.svg"))
        self.sidebar.add_tab("mining", "Mining", resource_path("gui/images/Mining.svg"))
        self.sidebar.add_tab("profile", "Profile", resource_path("gui/images/user.svg"))
        self.sidebar.tab_changed.connect(self._on_tab_changed)
        self.sidebar.connect_wallet_clicked.connect(self._on_connect_wallet)
        self.sidebar.user_guide_clicked.connect(self._show_user_guide)
        app_layout.addWidget(self.sidebar)

        content_wrapper = QWidget()
        content_wrapper.setObjectName("app_container")
        content_wrapper_layout = QVBoxLayout(content_wrapper)
        content_wrapper_layout.setContentsMargins(24, 24, 24, 24)

        self.screen_stack = QStackedWidget()
        self.wallet_screen = WalletScreen()
        self.wallet_screen.wallet_loaded.connect(self._on_wallet_loaded)
        self.wallet_screen.hotkey_imported.connect(self._on_hotkey_imported)
        self.mining_screen = MiningScreen(main_window=self)
        self.profile_screen = ProfileScreen()
        self.screen_stack.addWidget(self.wallet_screen)
        self.screen_stack.addWidget(self.mining_screen)
        self.screen_stack.addWidget(self.profile_screen)
        content_wrapper_layout.addWidget(self.screen_stack)

        app_layout.addWidget(content_wrapper, 1)

        self.content_stack.addWidget(app_container)

    def _apply_theme(self):
        self.setStyleSheet(BitSOTATheme.get_main_stylesheet())
        fonts = BitSOTATheme.get_font_system()
        self.setFont(fonts["primary"])

    def _on_start_clicked(self):
        self._show_user_guide()

    def _show_user_guide(self):
        guide_modal = UserGuideModal(parent=self)
        guide_modal.proceed_clicked.connect(self._on_user_guide_proceed)
        guide_modal.exec()

    def _on_user_guide_proceed(self):
        self.content_stack.setCurrentIndex(1)

    def _on_tab_changed(self, tab_id: str):
        if tab_id == "setup_wallet":
            self.screen_stack.setCurrentWidget(self.wallet_screen)
        elif tab_id == "mining":
            self.screen_stack.setCurrentWidget(self.mining_screen)
        elif tab_id == "profile":
            modal = ComingSoonModal(
                "Profile Screen",
                "The Profile screen is coming soon! This screen will show your mining history, rewards, and balances from both Direct Mining and Pool Mining. You'll be able to view detailed statistics and claim your rewards.",
                parent=self
            )
            modal.exec()
            self.sidebar.set_active_tab("mining")

    def _on_connect_wallet(self):
        self.sidebar.set_active_tab("setup_wallet")
        self.screen_stack.setCurrentWidget(self.wallet_screen)

    def _on_wallet_loaded(self, wallet_name: str, hotkey_name: str, use_existing_coldkey: bool, coldkey_address: str):
        from gui.wallet_utils_gui import (
            get_wallet_dir, get_bittensor_wallet_dir, discover_wallets,
            get_coldkey_address, save_coldkey_address, save_wallet_settings
        )

        wallet_dir = None
        wallets = discover_wallets()
        for w_name, hotkeys, source in wallets:
            if w_name == wallet_name and hotkey_name in hotkeys:
                if source == "bittensor":
                    wallet_dir = str(get_bittensor_wallet_dir())
                else:
                    wallet_dir = str(get_wallet_dir())
                break

        if not wallet_dir:
            wallet_dir = str(get_wallet_dir())

        self.wallet = Wallet(name=wallet_name, hotkey=hotkey_name, path=wallet_dir)

        try:
            hotkey = self.wallet.get_hotkey()
            address = hotkey.ss58_address
            short_address = f"{address[:6]}...{address[-4:]}" if address else "Unknown"
            self.sidebar.set_wallet_info(wallet_name, short_address)
        except Exception as e:
            print(f"Error loading hotkey: {e}")
            self.sidebar.set_wallet_info(wallet_name, "Error loading")
            return

        if use_existing_coldkey and coldkey_address:
            self.coldkey_address = coldkey_address
            save_coldkey_address(coldkey_address)
            save_wallet_settings(wallet_name, hotkey_name, coldkey_address)
            print(f"Using existing coldkey address: {coldkey_address}")
            short_coldkey = f"{coldkey_address[:6]}...{coldkey_address[-4:]}" if coldkey_address and len(coldkey_address) > 10 else coldkey_address
            self.sidebar.set_wallet_info(wallet_name, short_coldkey)

        self._initialize_client()
        self._update_mining_screen_status()

        if not use_existing_coldkey or not coldkey_address:
            self._prompt_for_coldkey_address()

    def _initialize_client(self):
        if not self.wallet:
            return

        self.contract_manager = None

        try:
            relay_endpoint = self._get_relay_endpoint_from_config()
            self.client = BittensorDirectClient(
                wallet=self.wallet,
                relay_endpoint=relay_endpoint,
                verbose=True,
                contract_manager=self.contract_manager
            )
            print(f"Direct client created successfully with relay: {relay_endpoint}")
        except Exception as e:
            print(f"Failed to create direct client: {e}")
            self.client = None

    @staticmethod
    def _get_relay_endpoint_from_config() -> str:
        return "https://relay.bitsota.com"

    def _update_mining_screen_status(self):
        if self.wallet and hasattr(self.mining_screen, 'update_wallet_status'):
            self.mining_screen.update_wallet_status(self.wallet.name)
            self.mining_screen.update_global_sota()

    def _prompt_for_coldkey_address(self):
        coldkey_modal = ColdkeyAddressModal(parent=self)
        coldkey_modal.address_submitted.connect(self._on_coldkey_address_submitted)
        coldkey_modal.exec()

    def _on_coldkey_address_submitted(self, address: str):
        from gui.wallet_utils_gui import save_coldkey_address

        self.coldkey_address = address
        save_coldkey_address(address)
        print(f"Coldkey address saved: {address}")

        short_address = f"{address[:6]}...{address[-4:]}" if address and len(address) > 10 else address
        if self.wallet:
            self.sidebar.set_wallet_info(self.wallet.name, short_address)

    def get_current_sota(self) -> Optional[float]:
        try:
            relay_endpoint = self._get_relay_endpoint_from_config()
            import requests
            response = requests.get(f"{relay_endpoint}/sota_threshold", timeout=10)
            response.raise_for_status()
            result = response.json()
            return result.get("sota_threshold")
        except Exception as e:
            print(f"Failed to fetch SOTA from relay: {e}")
            return None

    def _on_hotkey_imported(self, hotkey_name: str, mnemonic: str, coldkey_address: str):
        from gui.wallet_utils_gui import get_wallet_dir, save_wallet_settings
        from gui.components.import_confirmation_modals import ErrorModal
        import uuid

        wallet_name = f"imported_{str(uuid.uuid4())[:8]}"
        wallet_dir = str(get_wallet_dir())

        try:
            self.wallet = Wallet(name=wallet_name, hotkey=hotkey_name, path=wallet_dir)
            self.wallet.import_hotkey_from_mnemonic(mnemonic, overwrite=True)

            self.coldkey_address = coldkey_address if coldkey_address else None
            save_wallet_settings(wallet_name, hotkey_name, coldkey_address)

            hotkey = self.wallet.get_hotkey()
            address = hotkey.ss58_address
            short_address = f"{address[:6]}...{address[-4:]}" if address else "Unknown"
            self.sidebar.set_wallet_info(wallet_name, short_address)

            if not self.coldkey_address:
                self._prompt_for_coldkey_address()

            self._initialize_client()
            self._update_mining_screen_status()
        except Exception as e:
            error_modal = ErrorModal(
                "Import Failed",
                f"Failed to import hotkey. Please try again.\n\nError: {str(e)}",
                parent=self
            )
            error_modal.exec()
            print(f"Error importing hotkey: {e}")

    def _try_auto_load_wallet(self):
        from gui.wallet_utils_gui import (
            get_last_wallet, get_wallet_dir, get_bittensor_wallet_dir,
            discover_wallets, get_coldkey_address
        )

        last_wallet_name, last_hotkey_name = get_last_wallet()

        if not last_wallet_name or not last_hotkey_name:
            return

        wallets = discover_wallets()
        wallet_dir = None
        wallet_found = False

        for w_name, hotkeys, source in wallets:
            if w_name == last_wallet_name and last_hotkey_name in hotkeys:
                wallet_found = True
                if source == "bittensor":
                    wallet_dir = str(get_bittensor_wallet_dir())
                else:
                    wallet_dir = str(get_wallet_dir())
                break

        if not wallet_found:
            print(f"Last wallet {last_wallet_name}/{last_hotkey_name} not found")
            return

        if not wallet_dir:
            wallet_dir = str(get_wallet_dir())

        try:
            self.wallet = Wallet(name=last_wallet_name, hotkey=last_hotkey_name, path=wallet_dir)
            hotkey = self.wallet.get_hotkey()
            address = hotkey.ss58_address

            self.coldkey_address = get_coldkey_address()

            if self.coldkey_address:
                short_address = f"{self.coldkey_address[:6]}...{self.coldkey_address[-4:]}" if self.coldkey_address and len(self.coldkey_address) > 10 else self.coldkey_address
            else:
                short_address = f"{address[:6]}...{address[-4:]}" if address else "Unknown"

            self.sidebar.set_wallet_info(last_wallet_name, short_address)

            self._initialize_client()
            self._update_mining_screen_status()

            self.content_stack.setCurrentIndex(1)
            self.sidebar.set_active_tab("mining")
            self.screen_stack.setCurrentWidget(self.mining_screen)

            print(f"Auto-loaded wallet: {last_wallet_name}/{last_hotkey_name}")
        except Exception as e:
            print(f"Failed to auto-load wallet: {e}")

    def _setup_update_checker(self):
        QTimer.singleShot(2000, self._check_for_updates_on_startup)

        self.update_check_timer = QTimer()
        self.update_check_timer.timeout.connect(self._check_for_updates)
        self.update_check_timer.start(24 * 60 * 60 * 1000)

    def _check_for_updates_on_startup(self):
        print("[Main] Checking for updates on startup...")
        update_info = self.update_checker.check_for_updates(force=True)
        if update_info:
            print(f"[Main] Update found: {update_info}")
            self._show_update_modal(update_info)
        else:
            print("[Main] No updates available")

    def _check_for_updates(self):
        print("[Main] Periodic update check...")
        update_info = self.update_checker.check_for_updates()
        if update_info:
            print(f"[Main] Update found: {update_info}")
            self._show_update_modal(update_info)

    def _show_update_modal(self, update_info: dict):
        modal = UpdateAvailableModal(update_info, parent=self)
        modal.download_clicked.connect(lambda: self._download_update(update_info))
        modal.skip_clicked.connect(lambda: self._skip_update(update_info))
        modal.exec()

    def _download_update(self, update_info: dict):
        download_url = self.update_checker.get_download_url(update_info)
        if download_url:
            webbrowser.open(download_url)
            print(f"Opening download URL: {download_url}")
        else:
            print("No download URL available for this platform")

    def _skip_update(self, update_info: dict):
        self.update_checker.skip_version(update_info['new_version_code'])
        print(f"Skipped version {update_info['new_version']}")
