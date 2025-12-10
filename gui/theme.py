from PySide6.QtGui import QFont, QColor
from gui.resource_path import resource_path


class BitSOTATheme:
    """Design system based on Figma designs."""

    COLOR1 = "#150049"
    COLOR2 = "#8EFBFF"

    SECONDARY_BUTTON_BG = "#D0CCDB"

    START_SCREEN_BG = "#EDF1F1"

    APP_BG = "#F6F5FA"
    CONTENT_BOX_BG = "#FFFFFF"

    BORDER_12 = "rgba(21, 0, 73, 0.12)"
    BORDER_8 = "rgba(21, 0, 73, 0.08)"

    COLOR1_60 = "rgba(21, 0, 73, 0.60)"
    COLOR1_04 = "rgba(21, 0, 73, 0.04)"
    TAB_INACTIVE_BG = "rgba(109, 96, 142, 0.16)"

    @staticmethod
    def get_main_stylesheet() -> str:
        return f"""
        QMainWindow {{
            background-color: {BitSOTATheme.APP_BG};
            font-family: "Geist", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 16px;
            font-weight: 400;
        }}

        QWidget#start_screen {{
            background-color: {BitSOTATheme.START_SCREEN_BG};
        }}

        QWidget#sidebar {{
            background-color: {BitSOTATheme.APP_BG};
            border-right: 1px solid {BitSOTATheme.BORDER_12};
        }}

        QWidget#sidebar_logo_container {{
            background-color: transparent;
            border-bottom: 1px solid {BitSOTATheme.BORDER_12};
        }}

        QWidget#sidebar_tab {{
            background-color: transparent;
            border: none;
            border-radius: 8px;
        }}

        QWidget#sidebar_tab QLabel {{
            background-color: transparent;
            color: rgba(21, 0, 73, 0.60);
            border: none;
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 500;
            line-height: 20px;
        }}

        QWidget#sidebar_tab:hover {{
            background-color: rgba(21, 0, 73, 0.05);
        }}

        QWidget#sidebar_tab_active {{
            background-color: {BitSOTATheme.SECONDARY_BUTTON_BG};
            border: none;
            border-radius: 8px;
        }}

        QWidget#sidebar_tab_active QLabel {{
            background-color: transparent;
            color: #FFFFFF;
            border: none;
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 500;
            line-height: 20px;
        }}

        QLabel#sidebar_follow_label {{
            color: {BitSOTATheme.COLOR1};
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
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
            background-color: {BitSOTATheme.COLOR1_04};
            border: 1px solid {BitSOTATheme.BORDER_12};
            border-radius: 8px;
        }}

        QLabel#sidebar_wallet_name {{
            color: {BitSOTATheme.COLOR1};
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 500;
        }}

        QLabel#sidebar_wallet_address {{
            color: {BitSOTATheme.COLOR1_60};
            font-family: "JetBrains Mono", monospace;
            font-size: 12px;
            font-weight: 400;
        }}

        QPushButton#tab_switcher_active {{
            background-color: {BitSOTATheme.COLOR1_60};
            color: #FFFFFF;
            border: none;
            border-radius: 4px;
            padding: 4px 16px;
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
        }}

        QPushButton#tab_switcher_inactive {{
            background-color: {BitSOTATheme.TAB_INACTIVE_BG};
            color: {BitSOTATheme.COLOR1};
            border: none;
            border-radius: 4px;
            padding: 4px 16px;
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
        }}

        QLabel#mining_description {{
            color: {BitSOTATheme.COLOR1_60};
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
            line-height: 150%;
            letter-spacing: -0.42px;
        }}

        QLabel#section_title {{
            color: {BitSOTATheme.COLOR1};
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            font-weight: 500;
            line-height: 150%;
        }}

        QLabel#form_label {{
            color: {BitSOTATheme.COLOR1};
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 500;
        }}

        QComboBox#form_input {{
            background-color: {BitSOTATheme.CONTENT_BOX_BG};
            color: {BitSOTATheme.COLOR1};
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
            background-color: {BitSOTATheme.APP_BG};
            border: 1px solid {BitSOTATheme.BORDER_8};
            border-radius: 8px;
        }}

        QLabel#stat_label {{
            color: {BitSOTATheme.COLOR1};
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
            line-height: 150%;
        }}

        QLabel#stat_value {{
            color: {BitSOTATheme.COLOR1};
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
            line-height: 150%;
        }}

        QDialog#modal_dialog {{
            background-color: {BitSOTATheme.CONTENT_BOX_BG};
            border: 2px solid {BitSOTATheme.BORDER_8};
            border-radius: 12px;
        }}

        QLabel#modal_title {{
            color: {BitSOTATheme.COLOR1};
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 24px;
            font-weight: 500;
        }}

        QLabel#modal_message {{
            color: {BitSOTATheme.COLOR1};
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            font-weight: 400;
            line-height: 1.5;
        }}

        QPushButton#modal_close {{
            background-color: transparent;
            color: {BitSOTATheme.COLOR1};
            border: none;
            font-size: 24px;
            font-weight: 400;
        }}

        QLabel#metric_label {{
            color: {BitSOTATheme.COLOR1};
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
        }}

        QLabel#metric_value {{
            color: {BitSOTATheme.COLOR1};
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 32px;
            font-weight: 500;
        }}

        QLabel#info_icon {{
            color: {BitSOTATheme.COLOR1};
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
            background-color: {BitSOTATheme.CONTENT_BOX_BG};
            border: 1px solid {BitSOTATheme.BORDER_8};
            border-radius: 8px;
        }}

        QWidget#mining_config_box {{
            background-color: {BitSOTATheme.COLOR1_04};
            border-radius: 8px;
        }}

        QWidget#app_container {{
            background-color: {BitSOTATheme.APP_BG};
        }}

        QPushButton#primary_button {{
            background-color: {BitSOTATheme.COLOR1};
            color: {BitSOTATheme.COLOR2};
            border: none;
            border-radius: 4px;
            padding: 12px 32px;
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            font-weight: 400;
            line-height: 1.2;
            letter-spacing: -0.48px;
            text-transform: capitalize;
        }}

        QPushButton#primary_button:hover {{
            background-color: rgba(21, 0, 73, 0.9);
        }}

        QPushButton#primary_button:pressed {{
            background-color: rgba(21, 0, 73, 0.8);
        }}

        QPushButton#stop_mining_button {{
            background-color: {BitSOTATheme.COLOR1};
            color: #FF0000;
            border: none;
            border-radius: 4px;
            padding: 12px 32px;
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            font-weight: 400;
            line-height: 1.2;
            letter-spacing: -0.48px;
            text-transform: capitalize;
        }}

        QPushButton#stop_mining_button:hover {{
            background-color: rgba(21, 0, 73, 0.9);
        }}

        QPushButton#stop_mining_button:pressed {{
            background-color: rgba(21, 0, 73, 0.8);
        }}

        QPushButton#secondary_button {{
            background-color: {BitSOTATheme.SECONDARY_BUTTON_BG};
            color: {BitSOTATheme.COLOR1};
            border: none;
            border-radius: 4px;
            padding: 12px 32px;
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            font-weight: 400;
            line-height: 1.2;
            letter-spacing: -0.48px;
            text-transform: capitalize;
        }}

        QPushButton#secondary_button:hover {{
            background-color: rgba(208, 204, 219, 0.8);
        }}

        QPushButton#secondary_button:pressed {{
            background-color: rgba(208, 204, 219, 0.6);
        }}

        QPushButton#clear_logs_button {{
            background-color: {BitSOTATheme.COLOR1};
            color: #FFFFFF;
            opacity: 0.2;
            border: none;
            border-radius: 4px;
            padding: 4px 12px;
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
        }}

        QLabel {{
            color: {BitSOTATheme.COLOR1};
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        QLabel#start_tagline {{
            color: {BitSOTATheme.COLOR1};
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 20px;
            font-weight: 400;
            line-height: 1.2;
            letter-spacing: -0.6px;
            text-transform: capitalize;
        }}

        QLineEdit {{
            background-color: {BitSOTATheme.CONTENT_BOX_BG};
            color: {BitSOTATheme.COLOR1};
            border: 1px solid {BitSOTATheme.BORDER_12};
            border-radius: 4px;
            padding: 12px;
            font-size: 16px;
        }}

        QLineEdit:focus {{
            border: 1px solid {BitSOTATheme.COLOR1};
        }}

        QTextEdit {{
            background-color: {BitSOTATheme.CONTENT_BOX_BG};
            color: {BitSOTATheme.COLOR1};
            border: 1px solid {BitSOTATheme.BORDER_12};
            border-radius: 4px;
            padding: 12px;
            font-size: 16px;
        }}

        QTextEdit#logs_text {{
            background-color: {BitSOTATheme.CONTENT_BOX_BG};
            color: {BitSOTATheme.COLOR1};
            border: 1px solid {BitSOTATheme.BORDER_12};
            border-radius: 4px;
            padding: 12px;
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
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
            color: {BitSOTATheme.COLOR1};
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 18px;
            font-weight: 500;
            line-height: 150%;
        }}

        QLabel#wallet_option_desc {{
            color: {BitSOTATheme.COLOR1_60};
            font-family: "JetBrains Mono", monospace;
            font-size: 12px;
            font-weight: 400;
            line-height: 16px;
        }}

        QLabel#hotkey_credentials_title {{
            color: {BitSOTATheme.COLOR1};
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            font-weight: 500;
            line-height: 150%;
        }}

        QLineEdit#form_input {{
            background-color: {BitSOTATheme.CONTENT_BOX_BG};
            color: {BitSOTATheme.COLOR1};
            border: 1px solid {BitSOTATheme.BORDER_12};
            border-radius: 4px;
            padding: 8px 14px;
            font-size: 14px;
            height: 40px;
        }}

        QLineEdit#form_input:focus {{
            border: 1px solid {BitSOTATheme.COLOR1};
        }}

        QLineEdit#mnemonic_word_box {{
            background-color: {BitSOTATheme.COLOR1_04};
            border: 1px solid {BitSOTATheme.BORDER_12};
            border-radius: 4px;
            padding: 8px 12px;
            height: 68px;
            color: {BitSOTATheme.COLOR1_60};
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
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
            color: {BitSOTATheme.COLOR1};
            border: 1px solid {BitSOTATheme.BORDER_8};
            border-radius: 4px;
            padding: 16px;
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
            text-align: left;
        }}

        QPushButton#wallet_list_item:hover {{
            background-color: {BitSOTATheme.SECONDARY_BUTTON_BG};
            color: {BitSOTATheme.COLOR1};
        }}

        QPushButton#wallet_list_item_selected {{
            background-color: {BitSOTATheme.COLOR1};
            color: {BitSOTATheme.COLOR2};
            border: none;
            border-radius: 4px;
            padding: 16px;
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
            text-align: left;
        }}

        QCheckBox {{
            color: {BitSOTATheme.COLOR1};
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
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
            color: {BitSOTATheme.COLOR1};
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            font-weight: 500;
        }}

        QLabel#important_text {{
            color: {BitSOTATheme.COLOR1_60};
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
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
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            font-weight: 400;
        }}

        QPushButton#confirm_button_enabled {{
            background-color: {BitSOTATheme.COLOR1};
            color: {BitSOTATheme.COLOR2};
            border: none;
            border-radius: 4px;
            padding: 12px 32px;
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            font-weight: 400;
        }}

        QPushButton#confirm_button_enabled:hover {{
            background-color: rgba(21, 0, 73, 0.9);
        }}

        QLabel#success_title {{
            color: {BitSOTATheme.COLOR1};
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 24px;
            font-weight: 500;
        }}

        QLabel#success_message {{
            color: {BitSOTATheme.COLOR1};
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
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
