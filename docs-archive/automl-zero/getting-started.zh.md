# 快速开始

这个仓库包含多个服务与角色。请选择与你要做的事情相匹配的路径。

## 默认端口

- Relay: `http://127.0.0.1:8002`
- Sidecar: `http://127.0.0.1:8123`
- Pool API: `http://127.0.0.1:8434`
- Pool 监控面板: `http://127.0.0.1:9000`
- 文档网站: `http://127.0.0.1:9001`

## 我想运行文档网站

```bash
python3 -m venv .venv-docs
source .venv-docs/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements-docs.txt
mkdocs serve -a 127.0.0.1:9001
```

打开 `http://127.0.0.1:9001`。

## 我想跑一个本地端到端流程

参考 [本地测试](local-testing.md)。

## 我只想运行 relay

参考 [Relay](components/relay.md)。

## 我只想运行 Pool

参考 [Pool](components/pool.md)。
