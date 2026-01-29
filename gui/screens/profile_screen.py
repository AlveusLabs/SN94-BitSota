from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QScrollArea,
)

from .profile import (
    ClaimableRewardCard,
    SmallStatsCard,
    StakeUnlockCard,
    MiningHistoryTable,
)


PROFILE_SCREEN_STYLE = """
QScrollArea {
    background: transparent;
    border: none;
}
QScrollArea > QWidget > QWidget {
    background: transparent;
}
QLabel#page_title {
    font-size: 30px;
    font-weight: 600;
    color: #101828;
    letter-spacing: -0.75px;
}
"""


class ProfileScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        self.setStyleSheet(PROFILE_SCREEN_STYLE)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # ========== Scroll Area ==========
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)  # Content adapts to size
        scroll.setFrameShape(QFrame.Shape.NoFrame)  # No border
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.viewport().setStyleSheet("background: transparent;")
        
        # ========== Content Container ==========
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(24)

        # ========== Page Title ==========
        title = QLabel("Overview")
        title.setObjectName("page_title")
        content_layout.addWidget(title)
        
        # ========== Stats Cards Container ==========
        stats_container = QWidget()
        stats_layout = QHBoxLayout(stats_container)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(24)
        
        # ---------- Left Column (Claimable Reward + Two Small Cards) ----------
        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(24)
        
        # Claimable reward card
        self.claimable_card = ClaimableRewardCard()
        self.claimable_card.setMinimumHeight(192)
        self.claimable_card.claim_clicked.connect(self._on_claim_clicked)
        left_layout.addWidget(self.claimable_card)
        
        # Small cards row container
        small_cards_row = QWidget()
        small_cards_layout = QHBoxLayout(small_cards_row)
        small_cards_layout.setContentsMargins(0, 0, 0, 0)
        small_cards_layout.setSpacing(24)
        
        # Total TAO rewards card
        self.total_rewards_card = SmallStatsCard(
            "TOTAL $TAO REWARDS",
            icon_path="gui/images/profile/gift.svg"
        )
        self.total_rewards_card.setMinimumHeight(192)
        self.total_rewards_card.set_value("500.25", "$TAO")
        small_cards_layout.addWidget(self.total_rewards_card)
        
        # Cumulative runtime card (no dashed underline)
        self.runtime_card = SmallStatsCard(
            "CUMULATIVE RUNTIME",
            icon_path="gui/images/profile/clock.svg",
            show_dashed=False
        )
        self.runtime_card.setMinimumHeight(192)
        self.runtime_card.set_value("48h 20m", "")
        small_cards_layout.addWidget(self.runtime_card)
        
        left_layout.addWidget(small_cards_row)
        
        stats_layout.addWidget(left_column, 1)
        
        # ---------- Right Column (Stake & Unlock Rate Card) ----------
        self.stake_card = StakeUnlockCard()
        self.stake_card.setMinimumHeight(408)
        self.stake_card.stake_clicked.connect(self._on_stake_clicked)
        stats_layout.addWidget(self.stake_card, 1)
        
        content_layout.addWidget(stats_container)
        
        # ========== Mining History Table ==========
        self.mining_table = MiningHistoryTable()
        content_layout.addWidget(self.mining_table)
        
        content_layout.addStretch()
        
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
    
    def _on_claim_clicked(self):
        print("Claim clicked")
    
    def _on_stake_clicked(self):
        print("Stake clicked")
    
    def update_stats(self, claimable: str = "0.00", total_rewards: str = "0.00", 
                     runtime: str = "00h 00m"):
        """
        Update page displayed statistics
        
        Args:
            claimable: Claimable reward amount
            total_rewards: Total TAO reward amount
            runtime: Cumulative runtime
        """
        self.claimable_card.set_value(claimable)
        self.total_rewards_card.set_value(total_rewards, "$TAO")
        self.runtime_card.set_value(runtime, "")
