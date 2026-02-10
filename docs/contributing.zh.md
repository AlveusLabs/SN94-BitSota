# 贡献指南

## 仓库结构

- `relay/` 是 relay 服务
- `Pool/` 是 pool 服务
- `sidecar/` 是 GUI 使用的本地 API
- `miner/` 与 `neurons/` 包含 miner 入口
- `validator/` 与 `neurons/validator_node.py` 包含 validator 入口

## 修改文档

文档网站使用 MkDocs Material 构建。

```bash
python3 -m pip install -r requirements-docs.txt
mkdocs serve -a 127.0.0.1:9001
```
