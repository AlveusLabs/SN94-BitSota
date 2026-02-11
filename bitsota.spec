# -*- mode: python ; coding: utf-8 -*-
import os
import platform
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

torch_data = collect_data_files("torch")
torch_binaries = collect_dynamic_libs("torch")
numpy_data = collect_data_files("numpy")

datas = [
    ("gui/images", "gui/images"),
]
datas.extend(torch_data)
datas.extend(numpy_data)

hiddenimports = [
    "PIL",
    "numpy",
    "pandas",
    "torch",
    "torchvision",
    "PySide6",
    "fastapi",
    "uvicorn",
    "pydantic",
    "threading",
    "concurrent.futures",
    "_sqlite3",
    "sqlite3",
    "sqlite3.dbapi2",
    "multiprocessing",
    "multiprocessing.spawn",
    "multiprocessing.util",
    "multiprocessing.pool",
    "multiprocessing.queues",
]
hiddenimports.extend(collect_submodules("gui"))
hiddenimports.extend(collect_submodules("core"))
hiddenimports.extend(collect_submodules("miner"))
hiddenimports.extend(collect_submodules("neurons"))
hiddenimports.extend(collect_submodules("bittensor_network"))
hiddenimports.extend(collect_submodules("sidecar"))
hiddenimports.extend(collect_submodules("scripts"))
hiddenimports = list(dict.fromkeys(hiddenimports))

a = Analysis(
    [
        "gui/__main__.py",
        "scripts/bitsota_sidecar_entry.py",
        "scripts/miner_local_og_sidecar.py",
        "scripts/pool_miner_sidecar.py",
    ],
    pathex=[],
    binaries=torch_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)
program_scripts = a.scripts[-4:]
runtime_hook_scripts = a.scripts[:-4]

target_arch = None
if sys.platform == "darwin":
    build_arch = os.environ.get("BUILD_ARCH")
    if build_arch:
        target_arch = build_arch
    else:
        machine = platform.machine()
        target_arch = "arm64" if machine == "arm64" else "x86_64"

icon_file = None
if sys.platform == "darwin":
    icon_file = "app_icon.icns"
elif sys.platform == "win32":
    icon_file = "app_icon.ico"

worker_console = sys.platform != "win32"

exe_gui = EXE(
    pyz,
    runtime_hook_scripts + program_scripts[0:1],
    [],
    exclude_binaries=True,
    name="BitSota",
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

exe_sidecar = EXE(
    pyz,
    runtime_hook_scripts + program_scripts[1:2],
    [],
    exclude_binaries=True,
    name="BitSotaSidecar",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=worker_console,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=target_arch,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)

exe_miner = EXE(
    pyz,
    runtime_hook_scripts + program_scripts[2:3],
    [],
    exclude_binaries=True,
    name="BitSotaMiner",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=worker_console,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=target_arch,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)

exe_pool_miner = EXE(
    pyz,
    runtime_hook_scripts + program_scripts[3:4],
    [],
    exclude_binaries=True,
    name="BitSotaPoolMiner",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=worker_console,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=target_arch,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)

coll = COLLECT(
    exe_gui,
    exe_sidecar,
    exe_miner,
    exe_pool_miner,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="BitSota",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="BitSota.app",
        icon="app_icon.icns",
        bundle_identifier="com.bitsota.app",
        info_plist={
            "NSHighResolutionCapable": True,
            "CFBundleDisplayName": "BitSota",
            "CFBundleName": "BitSota",
            "LSUIElement": False,
        },
    )
