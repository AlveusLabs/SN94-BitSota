# Validator

验证者会：

- 从 autoresearch backend 请求已签名的 replay job
- 在 Docker/CUDA public replay sandbox 中重新评估分配到的提交
- 把观测到的指标提交回 backend
- 运行独立的 backend weight setter，应用 backend `validator_weights`

## 入口

- `python -m validator.research_validator_runner` 运行 replay validator
- `python -m validator.backend_weight_setter` 运行 backend 指定的链上权重设置

旧的 relay validator 与本地 winner 权重设置路径已经移除。请参考
[验证](../validation.md)、[Public Autoresearch Validator Runner](../public-validator-runner.md)
与 [配置参考](../configuration.md)。
