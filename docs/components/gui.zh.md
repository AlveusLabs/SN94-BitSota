# 桌面 GUI

GUI 会：

- 点击 Start Mining 后启动本地 sidecar 与 miner 进程
- 从 sidecar 读取日志与候选状态
- 将解提交到 relay

## 配置覆盖

本地与开发运行支持通过 JSON 覆盖端点配置。GUI 会按以下顺序读取，命中即止：

- `BITSOTA_GUI_CONFIG`  指向一个 JSON 文件的路径
- `./bitsota_gui_config.json`
- `./gui_config.json`
- `~/.bitsota/gui_config.json`

常用键：

- `relay_endpoint`
- `update_manifest_url`
- `pool_endpoint`
- `test_mode` 与 `test_invite_code`
- `problem_config_path`

端到端本地运行可从 [本地测试](../local-testing.md) 开始。
