# 故障排查

## Relay 数据库只读

显式指定一个你有权限的数据库路径：

```bash
python3 -m relay --test --database-url "sqlite:///./bitsota_relay_test.db"
```

## Sidecar 端口已被占用

在启动 GUI 前设置一个新端口：

```bash
export BITSOTA_SIDECAR_PORT=8124
```

## Pool 认证错误

Pool 端点需要 `X-Key`、`X-Timestamp` 与 `X-Signature`，并且时间戳必须在服务器时间 5 分钟范围内。

如果你在做手工测试，建议使用 `scripts/` 中已经生成认证头的脚本，而不是手工拼请求。
