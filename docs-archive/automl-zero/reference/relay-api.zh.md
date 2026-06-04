# Relay API

relay 是一个 FastAPI 服务。本地运行时，OpenAPI 页面在：

- `http://127.0.0.1:8002/docs`

每个响应都会包含 `X-Request-ID` 头，便于日志关联。

## 端点

| 方法 | 路径 | 认证 | 用途 |
|---|---|---|---|
| GET | `/health` | 无 | 存活检查 |
| GET | `/version.json` | 无 | GUI 更新清单 |
| GET | `/sota_threshold` | 无 | 当前 SOTA 阈值 |
| GET | `/sota-events` | 无 | 分页的公开 SOTA 事件 |
| POST | `/submit_solution` | miner | 提交或覆盖该矿工的最新解 |
| GET | `/results` | validator | 拉取最近矿工提交 |
| GET | `/results/{miner_hotkey}` | validator | 拉取某矿工最新提交 |
| POST | `/sota/vote` | validator | 投票最终确定 SOTA 事件 |
| GET | `/sota/events` | validator | 列出详细 SOTA 事件 |
| POST | `/blacklist/{miner_hotkey}` | validator | 投票将矿工加入黑名单 |
| POST | `/invitation_code/generate/{count}` | admin | 生成邀请码 |
| GET | `/invitation_code/list/{page}/{size}` | admin | 列出邀请码 |
| GET | `/invitation_code/linked` | miner | 获取与该矿工关联的邀请码 |
| POST | `/invitation_code/link` | miner | 将邀请码关联到该矿工 |
| POST | `/coldkey_address/update` | miner | 将 coldkey 与矿工关联 |
| POST | `/test/submit_solution` | validator uid0 | 仅用于日志的测试提交 |
| GET | `/test/submissions` | validator uid0 | 领取并获取排队的测试提交 |
| GET | `/admin/dashboard` | admin | 需要密码的 HTML 管理面板 |
| GET | `/admin/status` | admin | JSON 状态与请求速率指标 |

认证头：

- Miner 与 validator：`X-Key`, `X-Timestamp`, `X-Signature`
- Admin：`X-Auth-Token`
- Admin dashboard：HTTP Basic  默认 `admin` 与 `ADMIN_AUTH_TOKEN`  或 `X-Auth-Token`

## 示例

### 提交解

`POST /submit_solution`

```json
{
  "task_id": "cifar10_binary",
  "score": 0.93,
  "algorithm_result": {
    "dsl": "setup:\\n  CONST 0.5 -> s0\\n...",
    "metadata": {
      "engine": "baseline",
      "seed": 123
    }
  }
}
```

说明：
- 在非测试模式下，矿工必须先关联邀请码。
- relay 对每个矿工最多保留一个提交，并会在每次接受新提交时覆盖旧提交。

### 对候选投票

`POST /sota/vote`

```json
{
  "miner_hotkey": "5F...",
  "score": 0.931,
  "seen_block": 123456,
  "result_id": "abcd1234"
}
```

响应会返回：
- `status`，例如 `vote_recorded`、`vote_updated` 或 `finalized`
- 投票计数与当前 SOTA
- 仅当达成共识时才包含 `finalized_event`

## 常见错误

- `401` 缺少或无效的认证头
- `403` 矿工未受邀或已被拉黑
- `400` 分数缺失、低于 SOTA、在重复窗口内重复、或输入无效
 
## 权威实现

完整行为见 `relay/main.py`，请求与响应模型见 `relay/schemas.py`。
