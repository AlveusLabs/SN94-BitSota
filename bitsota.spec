# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import platform
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

torch_data = collect_data_files('torch')
torch_binaries = collect_dynamic_libs('torch')
numpy_data = collect_data_files('numpy')

datas = [
    ('gui/images'),
]

datas.extend(torch_data)
datas.extend(numpy_data)

a = Analysis(
    ['gui/__main__.py'],
    pathex=[],
    binaries=torch_binaries,
    datas=datas,
    hiddenimports=[
        'PIL',
        'numpy',
        'pandas',
        'torch',
        'torchvision',
        'PySide6',
        'PySide6.QtCore',
        'PySide6.QtWidgets',
        'PySide6.QtGui',
        'PySide6.QtSvg',
        'PySide6.QtSvgWidgets',
        'gui',
        'gui.main_window',
        'gui.theme',
        'gui.resource_path',
        'gui.wallet_utils_gui',
        'gui.components',
        'gui.components.sidebar',
        'gui.components.button',
        'gui.components.modal',
        'gui.components.wallet_selection_modal',
        'gui.components.import_confirmation_modals',
        'gui.components.tab_switcher',
        'gui.screens',
        'gui.screens.start_screen',
        'gui.screens.wallet_screen',
        'gui.screens.mining_screen',
        'gui.screens.pool_mining_screen',
        'gui.screens.profile_screen',
        'gui.sota_fetcher',
        'core',
        'core.algorithm',
        'core.task',
        'core.tasks',
        'miner',
        'miner.client',
        'miner.engines',
        'neurons',
        'neurons.miner',
        'bittensor_network',
        'bittensor_network.wallet',
        'common',
        'common.contract_manager',
        'validator',
        'threading',
        'concurrent.futures',
        '_sqlite3',
        'sqlite3',
        'sqlite3.dbapi2',
        'multiprocessing',
        'multiprocessing.spawn',
        'multiprocessing.util',
        'multiprocessing.pool',
        'multiprocessing.queues'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

target_arch = None
if sys.platform == 'darwin':
    machine = platform.machine()
    if machine == 'arm64':
        target_arch = 'arm64'
    else:
        target_arch = 'x86_64'

icon_file = None
if sys.platform == 'darwin':
    icon_file = 'app_icon.icns'
elif sys.platform == 'win32':
    icon_file = 'app_icon.ico'

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BitSota',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=target_arch,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BitSota',
)

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='BitSota.app',
        icon='app_icon.icns',
        bundle_identifier='com.bitsota.app',
        info_plist={
            'NSHighResolutionCapable': True,
            'CFBundleDisplayName': 'BitSota',
            'CFBundleName': 'BitSota',
            'LSUIElement': False,
        },
    )