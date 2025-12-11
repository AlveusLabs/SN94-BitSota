import os
import sys
import multiprocessing
from pathlib import Path
from datetime import datetime


def setup_logging():
    log_dir = Path.home() / ".bitsota" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    class Tee:
        def __init__(self, file, terminal):
            self.file = file
            # In a windowed app sys.stdout can be None; fall back to a devnull handle
            self.terminal = terminal

        def write(self, message):
            if self.terminal:
                try:
                    self.terminal.write(message)
                    self.terminal.flush()
                except Exception:
                    # If the terminal stream is not writable, ignore and keep logging to file
                    pass
            self.file.write(message)
            self.file.flush()

        def flush(self):
            if self.terminal:
                try:
                    self.terminal.flush()
                except Exception:
                    pass
            self.file.flush()

    log_handle = open(log_file, 'w', buffering=1)
    terminal = sys.stdout if sys.stdout is not None else open(os.devnull, 'w')
    sys.stdout = Tee(log_handle, terminal)
    sys.stderr = sys.stdout

    return log_file, log_handle


def main():
    log_file, log_handle = setup_logging()

    print("=" * 80)
    print("BitSota Starting...")
    print(f"Log file: {log_file}")
    print(f"Python: {sys.version}")
    print(f"Frozen: {getattr(sys, 'frozen', False)}")
    if getattr(sys, 'frozen', False):
        print(f"Bundle: {sys._MEIPASS}")
    print("=" * 80)

    try:
        print("\n[1/6] Setting multiprocessing start method...")
        multiprocessing.set_start_method('spawn', force=True)
        print("[2/6] Calling freeze_support...")
        multiprocessing.freeze_support()

        print("[3/6] Importing PySide6...")
        from PySide6.QtWidgets import QApplication, QMessageBox
        from PySide6.QtCore import QLockFile, QDir
        from gui.main_window import MiningWindow

        print("[4/6] Creating QApplication...")
        app = QApplication(sys.argv)
        app.setApplicationName("BitSota")
        app.setOrganizationName("BitSota")

        lock_file_path = QDir.temp().absoluteFilePath("bitsota.lock")
        lock_file = QLockFile(lock_file_path)

        if not lock_file.tryLock(100):
            print("Another instance already running!")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("BitSota Already Running")
            msg.setText("BitSota is already running. Only one instance can run at a time.")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            return 0

        print("[5/6] Creating main window...")
        window = MiningWindow()
        print("[6/6] Showing window and starting event loop...")
        window.show()
        result = app.exec()

        lock_file.unlock()
        print(f"\nApp exited normally with code: {result}")
        return result

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"FATAL ERROR: {e}")
        print(f"{'='*80}")
        import traceback
        traceback.print_exc()
        print(f"\nLog saved to: {log_file}")
        log_handle.flush()
        return 1
    finally:
        if 'log_handle' in locals():
            log_handle.flush()
            log_handle.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        emergency_log = Path.home() / ".bitsota" / "logs" / "emergency_crash.log"
        emergency_log.parent.mkdir(parents=True, exist_ok=True)
        with open(emergency_log, 'w') as f:
            import traceback
            f.write(f"Emergency crash before main(): {e}\n")
            traceback.print_exc(file=f)
        sys.exit(1)
