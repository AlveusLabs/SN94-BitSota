# 安全

## 密钥

- 不要分享助记词或私钥
- 冷钱包 coldkey 尽量离线保管
- 本地测试建议按角色拆分不同 hotkey

## API 认证

Relay 与 Pool 使用签名请求头进行认证。请把 hotkey 当作 API 凭证对待：

- 不要在公开日志中记录签名
- 生产环境不要在不同角色之间复用 hotkey

## Relay 管理员 token

- 把 `ADMIN_AUTH_TOKEN` 当作密码对待
- 一旦泄露请及时轮换

## 本地测试

- 使用专用的测试钱包与 hotkey
- 除非确实需要局域网访问，否则服务只绑定 localhost
