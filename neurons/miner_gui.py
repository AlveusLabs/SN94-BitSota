import sys
import multiprocessing


def main():
    multiprocessing.set_start_method('spawn', force=True)
    multiprocessing.freeze_support()

    from PySide6.QtWidgets import QApplication, QMessageBox
    from PySide6.QtCore import QLockFile, QDir
    from gui.main_window import MiningWindow

    app = QApplication(sys.argv)
    app.setApplicationName("BitSota")
    app.setOrganizationName("BitSota")

    lock_file_path = QDir.temp().absoluteFilePath("bitsota.lock")
    lock_file = QLockFile(lock_file_path)

    if not lock_file.tryLock(100):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("BitSota Already Running")
        msg.setText("BitSota is already running. Only one instance can run at a time.")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
        return 0

    window = MiningWindow()
    window.show()
    result = app.exec()

    lock_file.unlock()
    return result


if __name__ == "__main__":
    sys.exit(main())
