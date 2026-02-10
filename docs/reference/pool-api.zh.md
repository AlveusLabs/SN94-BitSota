# Pool API

Pool API 是一个 FastAPI 服务。本地运行时，OpenAPI 页面在：

- `http://127.0.0.1:8434/docs`

## 认证头

大多数矿工端点需要：

- `X-Key`  矿工 hotkey
- `X-Timestamp`  unix 秒
- `X-Signature`  对消息 `auth:{X-Timestamp}` 的 SR25519 签名

时间戳必须在服务器时间 5 分钟范围内。

## 端点

| 方法 | 路径 | 认证 | 用途 |
|---|---|---|---|
| GET | `/health` | 无 | 存活检查与当前 window 信息 |
| POST | `/api/v1/miners/register` | miner | 注册或刷新矿工 |
| GET | `/api/v1/miners/stats` | miner | 矿工统计与矿池摘要 |
| GET | `/api/v1/miners/leaderboard` | 无 | 按声誉排名的矿工榜 |
| GET | `/api/v1/tasks/pending_evaluations` | miner | 检查待处理评估分配 |
| POST | `/api/v1/tasks/request` | miner | 请求一个简单 evolve 或 evaluate 批次 |
| POST | `/api/v1/tasks/lease` | miner | 请求一个包含 evaluate、seed 与 gossip 的 lease |
| POST | `/api/v1/tasks/{lease_id}/submit_lease` | miner | 提交 lease 结果包 |
| POST | `/api/v1/tasks/{batch_id}/submit_evolution` | miner | 为 batch 提交演化算法 |
| POST | `/api/v1/tasks/{batch_id}/submit_evaluation` | miner | 为 batch 提交评估结果 |
| GET | `/api/v1/results` | miner | 列出已验证 results |
| GET | `/api/v1/results/sota` | 无 | 某任务类型的当前 pool SOTA |
| POST | `/api/v1/results/verify/{result_id}` | miner | 将已完成 result 标记为已验证 |
| GET | `/api/v1/monitor/summary` | monitor | 聚合的监控摘要 |

## 示例

### 注册

`POST /api/v1/miners/register` 仅需认证头。响应会包含当前声誉与时间戳等信息。

### Lease 工作

请求一个 bundle：

`POST /api/v1/tasks/lease`

```json
{
  "task_type": "cifar10_binary",
  "eval_batch_size": 8,
  "seed_batch_size": 2,
  "gossip_limit": 5
}
```

提交一个 bundle：

`POST /api/v1/tasks/{lease_id}/submit_lease`

```json
{
  "evaluations": [],
  "evolutions": [],
  "gossip": null
}
```

## 权威实现

请求处理见 `Pool/app/api/v1`，模型见 `Pool/app/schemas`。
