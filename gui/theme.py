from PySide6.QtGui import QFont, QColor
from gui.resource_path import resource_path


class BitSOTATheme:
    """Design system based on Figma designs."""

    # Brand Colors
    COLOR1 = "#150049"  # Brand - Dark Color
    COLOR2 = "#8EFBFF"  # bitsota Color2
    COLOR2_VARIANT = "#71DADE"  # Color2 variant for button text

    # Dark Theme Colors
    BLACK100 = "#0C0029"  # Dark background
    BLACK60 = "rgba(12, 0, 41, 0.6)"  # 60% dark
    
    # Main Color Variants
    COLOR1_60 = "rgba(21, 0, 73, 0.6)"  # Primary color 60%
    COLOR1_12 = "rgba(21, 0, 73, 0.12)"  # Primary color 12%
    COLOR1_08 = "rgba(21, 0, 73, 0.08)"  # Primary color 8%
    COLOR1_04 = "rgba(21, 0, 73, 0.04)"  # Primary color 4%
    COLOR1_20 = "rgba(21, 0, 73, 0.2)"  # Primary color 20% (placeholder)

    SECONDARY_BUTTON_BG = "#D0CCDB"
    
    # Background Colors - Updated to Dark Theme
    APP_BG = "#0C0029"  # Changed to dark background
    CONTENT_BOX_BG = "#FFFFFF"  # White content area
    START_SCREEN_BG = "#0C0029"  # Changed to dark

    # Border Colors
    BORDER_12 = "rgba(21, 0, 73, 0.12)"
    BORDER_8 = "rgba(21, 0, 73, 0.08)"
    BORDER_4 = "#A199B6"
    
    # Legacy compatibility
    TAB_INACTIVE_BG = "rgba(109, 96, 142, 0.16)"
    
    # Radius
    RADIUS_4 = "4px"

    @staticmethod
    def get_main_stylesheet() -> str:
        return f"""
        QMainWindow {{
            background-color: {BitSOTATheme.BLACK100};
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 16px;
            font-weight: 400;
        }}

        QWidget#start_screen {{
            background-color: {BitSOTATheme.CONTENT_BOX_BG};
        }}

        /* Top navigation bar */
        QWidget#topbar {{
            background-color: {BitSOTATheme.BLACK100};
            border-bottom: none;
        }}

        QWidget#nav_tab {{
            background-color: transparent;
            padding: 0px;
        }}

        QLabel#nav_tab_label {{
            color: rgba(255, 255, 255, 0.6);
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 500;
            line-height: 20px;
            padding: 0px 10px;
        }}

        QWidget#nav_tab_indicator {{
            background-color: #FFFFFF;
            border-radius: 100px;
        }}

        QWidget#icon_button {{
            background-color: transparent;
            border-radius: 4px;
        }}

        QWidget#icon_button:hover {{
            background-color: rgba(255, 255, 255, 0.1);
        }}

        QWidget#wallet_dropdown {{
            background-color: transparent;
            border-radius: 4px;
        }}

        QWidget#wallet_dropdown:hover {{
            background-color: rgba(255, 255, 255, 0.16);
        }}

        QLabel#wallet_name_label {{
            color: #FFFFFF;
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            font-weight: 500;
            line-height: 14px;
        }}

        QWidget#wallet_not_connected_button {{
            background-color: transparent;
            border-radius: 4px;
        }}

        QWidget#wallet_not_connected_button:hover {{
            background-color: rgba(255, 255, 255, 0.05);
        }}

        QLabel#wallet_not_connected_label {{
            color: rgba(255, 255, 255, 0.8);
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 600;
            line-height: 1.2;
        }}

        /* Keep sidebar styles for compatibility */
        QWidget#sidebar {{
            background-color: {BitSOTATheme.BLACK100};
            border-right: none;
        }}

        QWidget#sidebar_logo_container {{
            background-color: transparent;
            border-bottom: none;
        }}

        QWidget#sidebar_tab {{
            background-color: transparent;
            border: none;
            border-radius: 8px;
        }}

        QWidget#sidebar_tab QLabel {{
            background-color: transparent;
            color: rgba(255, 255, 255, 0.6);
            border: none;
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 500;
            line-height: 20px;
        }}

        QWidget#sidebar_tab:hover {{
            background-color: rgba(255, 255, 255, 0.1);
        }}

        QWidget#sidebar_tab_active {{
            background-color: transparent;
            border: none;
            border-bottom: 2px solid #FFFFFF;
            border-radius: 0px;
        }}

        QWidget#sidebar_tab_active QLabel {{
            background-color: transparent;
            color: #FFFFFF;
            border: none;
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 500;
            line-height: 20px;
        }}

        QLabel#sidebar_follow_label {{
            color: #FFFFFF;
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 500;
        }}

        QWidget#social_icon {{
            background-color: transparent;
            border: none;
            border-radius: 4px;
        }}

        QWidget#social_icon:hover {{
            background-color: rgba(21, 0, 73, 0.05);
        }}

        QWidget#sidebar_wallet_info {{
            background-color: transparent;
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 4px;
        }}

        QLabel#sidebar_wallet_name {{
            color: #FFFFFF;
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            font-weight: 500;
        }}

        QLabel#sidebar_wallet_address {{
            color: rgba(255, 255, 255, 0.8);
            font-family: "JetBrains Mono", monospace;
            font-size: 12px;
            font-weight: 400;
        }}

        QWidget#tab_switcher_container {{
            background-color: #F6F5F8;
            border-radius: 4px;
            padding: 4px;
        }}

        QPushButton#tab_switcher_active {{
            height: 32px;
            padding: 0px 16px;
            background-color: #FFFFFF;
            color: {BitSOTATheme.BLACK100};
            border: none;
            border-radius: 4px;
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 500;
        }}

        QPushButton#tab_switcher_inactive {{
            height: 32px;
            padding: 0px 16px;
            background-color: transparent;
            color: rgba(12, 0, 41, 0.6);
            border: none;
            border-radius: 4px;
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 500;
        }}

        QLabel#mining_description {{
            color: rgba(12, 0, 41, 0.6);
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
            line-height: 150%;
            letter-spacing: -0.42px;
        }}

        QLabel#section_title {{
            color: {BitSOTATheme.BLACK100};
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 32px;
            font-weight: 600;
            line-height: 150%;
        }}

        QLabel#config_section_title {{
            color: {BitSOTATheme.BLACK100};
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 20px;
            font-weight: 500;
            line-height: 150%;
        }}

        QLabel#logs_title {{
            color: {BitSOTATheme.BLACK100};
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            font-weight: 500;
            line-height: 150%;
        }}

        QLabel#form_label {{
            color: {BitSOTATheme.BLACK60};
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 500;
        }}

        QComboBox#form_input {{
            background-color: {BitSOTATheme.CONTENT_BOX_BG};
            color: {BitSOTATheme.BLACK100};
            border: 1px solid {BitSOTATheme.BORDER_12};
            border-radius: 4px;
            padding: 8px 14px;
            font-size: 14px;
            min-height: 48px;
        }}

        QComboBox#form_input::drop-down {{
            border: none;
            width: 30px;
        }}

        QComboBox#form_input::down-arrow {{
            image: url({resource_path("gui/images/chevron-down-2.svg")});
            width: 16px;
            height: 16px;
        }}

        QComboBox#form_input QAbstractItemView {{
            background-color: {BitSOTATheme.CONTENT_BOX_BG};
            border: 1px solid {BitSOTATheme.BORDER_12};
            selection-background-color: {BitSOTATheme.TAB_INACTIVE_BG};
            padding: 4px;
        }}

        QWidget#stats_box {{
            background-color: {BitSOTATheme.COLOR1_04};
            border-radius: 4px;
        }}

        QWidget#logs_box {{
            background-color: {BitSOTATheme.COLOR1_04};
            border-radius: 4px;
        }}

        QWidget#stat_divider {{
            background-color: rgba(21, 0, 73, 0.12);
        }}

        QLabel#stat_label {{
            color: rgba(12, 0, 41, 0.4);
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
        }}

        QLabel#stat_value {{
            color: {BitSOTATheme.BLACK100};
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
        }}

        /* Status indicator - Running */
        QWidget#status_dot_running {{
            background-color: #1A7544;
            border-radius: 3px;
        }}

        QLabel#status_text_running {{
            color: #1A7544;
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
        }}

        /* Status indicator - Connected */
        QWidget#status_dot_connected {{
            background-color: #158047;
            border-radius: 3px;
        }}

        QLabel#status_text_connected {{
            color: #158047;
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
        }}

        /* Status indicator - Idle/Disconnected */
        QWidget#status_dot_idle, QWidget#status_dot_disconnected {{
            background-color: rgba(12, 0, 41, 0.4);
            border-radius: 3px;
        }}

        QLabel#status_text_idle, QLabel#status_text_disconnected {{
            color: rgba(12, 0, 41, 0.4);
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
        }}

        /* Modal dialog styles moved to respective modal component files */
        /* See UserGuideModal.get_stylesheet(), ColdkeyAddressModal.get_stylesheet(), etc. */

        QPushButton#modal_close {{
            background-color: transparent;
            color: {BitSOTATheme.BLACK100};
            border: none;
            font-size: 24px;
            font-weight: 400;
        }}

        QLabel#metric_label {{
            color: {BitSOTATheme.BLACK100};
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
        }}

        QLabel#metric_value {{
            color: {BitSOTATheme.BLACK100};
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 32px;
            font-weight: 500;
        }}

        QLabel#info_icon {{
            color: {BitSOTATheme.BLACK100};
            font-size: 14px;
        }}

        QProgressBar#pool_progress {{
            background-color: rgba(21, 0, 73, 0.1);
            border: none;
            border-radius: 4px;
        }}

        QProgressBar#pool_progress::chunk {{
            background-color: {BitSOTATheme.COLOR1};
            border-radius: 4px;
        }}

        QWidget#content_box {{
            background-color: #FFFFFF;
            border-radius: 4px;
        }}

        QWidget#mining_config_box {{
            background-color: {BitSOTATheme.COLOR1_04};
            border-radius: 4px;
        }}

        QWidget#app_container {{
            background-color: #FFFFFF;
        }}
        
        QWidget#content_wrapper {{
            background-color: {BitSOTATheme.BLACK100};
        }}
        
        QStackedWidget#screen_stack {{
            background-color: {BitSOTATheme.CONTENT_BOX_BG};
            padding: 32px;
            border-radius: 4px;
        }}

        /* Primary button styles moved to gui/components/common/button.py */
        /* See PrimaryButton.get_stylesheet() for button styles */

        QPushButton#stop_mining_button {{
            background-color: #EF4858;
            color: #FFFFFF;
            border: none;
            border-radius: 4px;
            padding: 20px 10px;
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            font-weight: 600;
            line-height: 1.2;
            text-transform: capitalize;
        }}

        QPushButton#stop_mining_button:hover {{
            background-color: #E03848;
        }}

        QPushButton#stop_mining_button:pressed {{
            background-color: #D02838;
        }}

        QPushButton#stop_mining_button QWidget#icon_container {{
            background-color: transparent;
            border: none;
        }}

        QPushButton#stop_mining_button QLabel#button_text_label {{
            background: transparent;
            border: none;
            color: #FFFFFF;
        }}

        QPushButton#clear_logs_button {{
            background-color: #D0CCDB;
            color: {BitSOTATheme.BLACK100};
            border: none;
            border-radius: 4px;
            padding: 10px 24px;
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 600;
        }}

        QLabel {{
            color: {BitSOTATheme.BLACK100};
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        QLabel#start_tagline {{
            color: #150049;
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 20px;
            font-weight: 400;
            line-height: 1.2;
            letter-spacing: -0.6px;
            text-transform: capitalize;
        }}

        QLineEdit {{
            background-color: {BitSOTATheme.CONTENT_BOX_BG};
            color: {BitSOTATheme.BLACK100};
            border: 1px solid {BitSOTATheme.BORDER_12};
            border-radius: 4px;
            padding: 12px 14px;
            font-size: 14px;
            min-height: 48px;
        }}

        QLineEdit:focus {{
            border: 1px solid {BitSOTATheme.COLOR1};
        }}

        QTextEdit {{
            background-color: {BitSOTATheme.CONTENT_BOX_BG};
            color: {BitSOTATheme.BLACK100};
            border: 1px solid {BitSOTATheme.BORDER_12};
            border-radius: 4px;
            padding: 12px;
            font-size: 16px;
        }}

        QTextEdit#logs_text {{
            background-color: transparent;
            color: rgba(12, 0, 41, 0.6);
            border: none;
            padding: 0px;
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            line-height: 150%;
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 500;
            line-height: 150%;
            letter-spacing: -0.42px;
            opacity: 0.6;
        }}

        QScrollBar:vertical {{
            background-color: transparent;
            width: 12px;
            margin: 0px;
        }}

        QScrollBar::handle:vertical {{
            background-color: {BitSOTATheme.BORDER_12};
            border-radius: 6px;
            min-height: 25px;
        }}

        QScrollBar::handle:vertical:hover {{
            background-color: {BitSOTATheme.COLOR1};
        }}

        QWidget#wallet_option_container {{
            background-color: {BitSOTATheme.CONTENT_BOX_BG};
            border: 1px solid {BitSOTATheme.BORDER_8};
            border-radius: 8px;
        }}

        QWidget#wallet_option_container:hover {{
            border: 2px solid {BitSOTATheme.BORDER_12};
        }}

        QLabel#wallet_option_title {{
            color: {BitSOTATheme.BLACK100};
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 18px;
            font-weight: 500;
            line-height: 150%;
        }}

        QLabel#wallet_option_desc {{
            color: {BitSOTATheme.BLACK60};
            font-family: "JetBrains Mono", monospace;
            font-size: 12px;
            font-weight: 400;
            line-height: 16px;
        }}

        QLabel#hotkey_credentials_title {{
            color: {BitSOTATheme.BLACK100};
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 24px;
            font-weight: 600;
            line-height: 150%;
        }}

        QLineEdit#form_input {{
            background-color: {BitSOTATheme.CONTENT_BOX_BG};
            color: {BitSOTATheme.BLACK100};
            border: 1px solid {BitSOTATheme.BORDER_12};
            border-radius: 4px;
            padding: 8px 14px;
            font-size: 14px;
            min-height: 48px;
        }}

        QLineEdit#form_input:focus {{
            border: 1px solid {BitSOTATheme.COLOR1};
        }}

        QLineEdit#form_input::placeholder {{
            color: {BitSOTATheme.COLOR1_20};
        }}

        QLineEdit#mnemonic_word_box {{
            background-color: {BitSOTATheme.COLOR1_04};
            border: 1px solid {BitSOTATheme.BORDER_12};
            border-radius: 4px;
            padding: 8px 12px;
            height: 68px;
            color: {BitSOTATheme.BLACK60};
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
            line-height: 150%;
            letter-spacing: -0.42px;
            text-transform: capitalize;
            text-align: center;
        }}

        QLineEdit#mnemonic_word_box:focus {{
            border: 1px solid {BitSOTATheme.COLOR1};
        }}

        QPushButton#wallet_list_item {{
            background-color: {BitSOTATheme.CONTENT_BOX_BG};
            color: {BitSOTATheme.BLACK100};
            border: 1px solid {BitSOTATheme.BORDER_8};
            border-radius: 4px;
            padding: 16px;
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
            text-align: left;
        }}

        QPushButton#wallet_list_item:hover {{
            background-color: {BitSOTATheme.SECONDARY_BUTTON_BG};
            color: {BitSOTATheme.BLACK100};
        }}

        QPushButton#wallet_list_item_selected {{
            background-color: {BitSOTATheme.COLOR1};
            color: {BitSOTATheme.COLOR2};
            border: none;
            border-radius: 4px;
            padding: 16px;
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
            text-align: left;
        }}

        QCheckBox {{
            color: {BitSOTATheme.BLACK100};
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
            spacing: 8px;
        }}

        QCheckBox::indicator {{
            width: 20px;
            height: 20px;
            border: 1px solid {BitSOTATheme.BORDER_12};
            border-radius: 4px;
            background-color: {BitSOTATheme.CONTENT_BOX_BG};
        }}

        QCheckBox::indicator:checked {{
            background-color: {BitSOTATheme.COLOR1};
            border: 1px solid {BitSOTATheme.COLOR1};
            image: url({resource_path("gui/images/tick.svg")});
        }}

        QCheckBox::indicator:hover {{
            border: 1px solid {BitSOTATheme.COLOR1};
        }}

        QLabel#important_title {{
            color: {BitSOTATheme.BLACK100};
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            font-weight: 500;
        }}

        QLabel#important_text {{
            color: {BitSOTATheme.BLACK60};
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
            line-height: 150%;
        }}

        QPushButton#confirm_button_disabled {{
            background-color: rgba(208, 204, 219, 0.5);
            color: rgba(21, 0, 73, 0.4);
            border: none;
            border-radius: 4px;
            padding: 12px 32px;
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            font-weight: 400;
        }}

        QPushButton#confirm_button_enabled {{
            background-color: {BitSOTATheme.COLOR1};
            color: {BitSOTATheme.COLOR2};
            border: none;
            border-radius: 4px;
            padding: 12px 32px;
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            font-weight: 400;
        }}

        QPushButton#confirm_button_enabled:hover {{
            background-color: rgba(21, 0, 73, 0.9);
        }}

        QLabel#success_title {{
            color: {BitSOTATheme.BLACK100};
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 24px;
            font-weight: 600;
        }}

        QLabel#success_message {{
            color: {BitSOTATheme.BLACK100};
            font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            font-weight: 400;
            line-height: 150%;
        }}
        """

    @staticmethod
    def get_font_system():
        fonts = {}

        primary_font = QFont()
        primary_font.setFamilies([
            "PingFang SC",
            "Microsoft YaHei",
            "Geist",
            "-apple-system",
            "BlinkMacSystemFont",
            "Segoe UI",
            "Roboto",
            "sans-serif",
        ])
        primary_font.setPointSize(16)
        primary_font.setWeight(QFont.Weight.Normal)
        fonts["primary"] = primary_font

        return fonts
