# 配置参考

- Miner 配置：`miner_config.yaml`
- Validator 配置：`validator_config.yaml`
- Relay 配置：环境变量与 `python -m relay` 参数
- 桌面 GUI 开发覆盖：`new_gui` 的 JSON 配置与环境变量

如果你想了解奖励机制，请看 `docs/reward-modes.md`。如果你想了解如何运行矿工或验证者，请看 `docs/mining.md`、`docs/pool-mining.md` 与 `docs/validation.md`。

---

## Miner  `miner_config.yaml`

使用方：
- CLI miner：`neurons/miner.py`

### `wallet`

- `wallet.wallet_name`  字符串 必填：Bittensor 钱包名 也就是 `wallet.path` 下的目录名
- `wallet.hotkey_name`  字符串 必填：钱包内的 hotkey 名称

### `pool_url`

- `pool_url`  字符串或 `null`：
  - `null` → **直接模式** 提交到 relay 与 validators
  - 非空 URL → **矿池模式** 与 pool 服务通信

### `validators`

- `validators`  list[string]：relay 的 base URL 列表。在直接模式下，CLI 使用 **第一个** 作为 relay endpoint。如果为空，miner 会回退到 `http://127.0.0.1:8002`。

### `mining`

- `mining.mode`  字符串：UI 与元信息提示值 取 `direct` 或 `pool`。当前 CLI 是否走直接或矿池，严格由 `pool_url` 决定。
- `mining.task_type`  字符串：任务标识。
  - 默认子网任务为 `cifar10_binary`。在 `test_mode` 本地测试中，GUI 也会暴露 `mnist_binary` 与 `scalar_linear`。
- `mining.engine_type`  字符串：直接模式下的演化引擎 取 `archive` 或 `ga`。
- `mining.cycles`  int：仅矿池模式使用，`0` 表示一直运行。
- `mining.alternate_tasks`  bool：仅矿池模式使用，在 `evolve` 与 `evaluate` 任务请求之间交替。
- `mining.delay`  float 秒：仅矿池模式使用，每个 cycle 之间 sleep 的时长。
- `mining.max_retries`  int：仅矿池模式使用，pool 任务请求的重试次数。代码已支持，如需使用可自行加入到 YAML。默认 `3`。
- `mining.iterations`  int：保留字段 旧字段  `neurons/miner.py` 不使用。
- `mining.verbose`  bool：保留字段 旧字段。当前直接客户端的详细日志由 `evolution.verbose` 驱动。

### `evolution`

- `evolution.max_generations`  int：仅直接模式使用，每次挖矿循环的代数。CLI 会用该值设置 `MAX_EVOLUTION_GENERATIONS`。
- `evolution.verbose`  bool：启用额外 miner 日志。
- `evolution.fec`  dict 可选：Functional Equivalence Cache  FEC 配置。
  - `evolution.fec.cache_size`  int：基于 probe 的 FEC 的 LRU 缓存大小，默认 `100000`，`0` 表示禁用。
  - `evolution.fec.num_train_examples`  int：probe 训练子集大小，默认 `32`。
  - `evolution.fec.num_valid_examples`  int：probe 验证子集大小，默认 `32`。
  - `evolution.fec.forget_every`  int：每插入 N 次清空缓存，默认 `0` 表示禁用。

### `logging`

- `logging.level`  字符串：Python 日志级别 `DEBUG`、`INFO`、`WARNING`、`ERROR`。
- `logging.file`  字符串或 `null`：如果设置，则把日志写到此文件。目录会自动创建。

## Validator  `validator_config.yaml`

使用方：`neurons/validator_node.py` 以及 `bittensor_network/*` 辅助模块。

### `reward_mode`

- `reward_mode`  字符串：
  - `capacitor` → 通过 EVM 合约投票提交奖励  `ContractManager`
  - `capacitorless` → relay SOTA 投票 加 链上权重
  - `capacitorless_sticky` → relay 或本地 winner 加 sticky 的 burn split 权重

行为差异见 `docs/reward-modes.md`。

### Bittensor 网络设置

- `netuid`  int：子网 netuid。
- `wallet_name`  字符串：钱包名。
- `wallet_hotkey`  字符串：hotkey 名称。
- `path`  字符串：钱包根路径 例如 `~/.bittensor/wallets/`。
- `network`  字符串：subtensor 网络名 例如 `test`、`finney`。
- `subtensor_chain_endpoint`  字符串或 `null`：覆盖 websocket endpoint，例如 `wss://test.finney.opentensor.ai:443`。`null` 让 bittensor 自行选择。
- `epoch_length`  int：本地用作权重更新的最小区块间隔门限，并与链上限速对比。

### Capacitor  EVM 合约 设置

仅当 `reward_mode: capacitor` 时使用。

- `evm_key_path`  字符串或 `null`：包含 `{\"private_key\": \"0x...\"}` 的 EVM key JSON 文件路径。
  - 如果省略，validator 会回退到 `EVM_PRIVATE_KEY` 或 bittensor 钱包的 `h160/` key 文件。
- `contract.rpc_url`  字符串：EVM JSON RPC endpoint。
- `contract.address`  字符串：合约地址。
- `contract.abi_file`  字符串：ABI JSON 路径，默认 `capacitor_abi.json`。

### Capacitorless 设置  `reward_mode: capacitorless*`

以下键都在 `capacitorless:` 下。

- `capacitorless.mode`  字符串：
  - `sticky_burnsplit` 默认行为 → burn 加 winner 分成，winner 会保持直到被替换
  - `windowed` → 仅在 relay 奖励窗口内把全部权重给 winner，否则为 burn
- `capacitorless.burn_hotkey`  字符串 必填：接收 burn emissions 的已注册 hotkey。
- `capacitorless.burn_share`  float：用于 `sticky_burnsplit`，默认 `0.9`。
- `capacitorless.winner_share`  float 或省略：用于 `sticky_burnsplit`，默认 `1 - burn_share`。如果总和不为 1，会做归一化。
- `capacitorless.winner_source`  字符串：`relay` 或 `local`，仅 sticky 模式使用。
- `capacitorless.min_winner_improvement`  float：仅本地 winner 模式使用，替换当前本地 winner 的最小提升幅度。
- `capacitorless.submit_sota_votes`  bool：若为 `false`，不向 relay 发送 `/sota/vote`，用于纯本地 capacitorless 运行。
- `capacitorless.apply_weights_inline`  bool：仅本地 winner 模式使用，若为 `true`，在评估后立即应用权重变化，而不是等待后台循环。
- `capacitorless.alignment_mod`  int：仅 windowed 模式使用，对齐权重更新的区块间隔，默认 `360`。
- `capacitorless.events_limit`  int：权重调度抓取 relay events 的上限。
- `capacitorless.event_refresh_interval_s`  int：刷新 relay events 的频率，sticky 模式使用。
- `capacitorless.metagraph_refresh_interval_s`  int：权重循环刷新 metagraph 的频率。
- `capacitorless.poll_interval_s`  float：权重循环的轮询间隔。
- `capacitorless.retry_interval_s`  float：两次尝试应用权重的最小间隔秒数。

### Relay 轮询

- `relay.url`  字符串：relay base URL。用于 relay 轮询时必填，capacitroless 模式也必填。
- `relay.poll_interval_seconds`  int：抓取新提交的轮询间隔。

### 提交调度  可选限流

- `submission_schedule.mode`  字符串：`immediate`、`interval` 或 `utc_times`。
- `submission_schedule.interval_seconds`  int：当 mode 为 `interval` 时使用。
- `submission_schedule.utc_times`  list[string]：当 mode 为 `utc_times` 时使用，例如 `\"00:00\"`  以 UTC 为准。

### 提交阈值门

- `submission_threshold.mode`  字符串：`sota_only` 或 `local_best`。
  - `local_best` 还要求候选必须超过该 validator 进程生命周期内看到的本地最佳分数。

### 黑名单策略

- `blacklist.cutoff_percentage`  float：矿工声称分数与 validator 分数之间允许的最大 **绝对** 差值，超过则投票拉黑。分数在 `[0, 1]` 时，`0.1` 约等于 10%。

### SOTA 回退

- `sota.cache_duration`  int 秒：保留字段。当前 validator 在内部缓存 SOTA，relay 也会单独缓存。
- `sota.default_threshold`  float：当无法从 relay 或合约获取 SOTA 时使用的默认阈值。

### 权重

- `weights.wait_for_inclusion`  bool：通过 `bittensor_network/_weights.py` 传给 `subtensor.set_weights(...)`。
- `weights.wait_for_finalization`  bool：通过 `bittensor_network/_weights.py` 传给 `subtensor.set_weights(...)`。
- `weights.check_interval`  int 秒：保留字段，预期用于控制权重后台循环运行频率。
- `weights.auto_restart`  bool：保留字段，预期用于权重循环崩溃后自动重启。

### 可选  `contract_bots`

- `contract_bots`  list[string]：在 `reward_mode: capacitor` 中由 `WeightManager` 使用，用于对固定一组 hotkey 设置权重。

---

## 超参数 JSON  推荐

miner 与 validator 的评估默认值现在来自仓库根目录的 JSON 文件：

- `miner_hyperparams.json`  直接挖矿 miner 默认值
- `validator_hyperparams.json`  validator 评估默认值

这用于替代日常调参时通过大量环境变量控制的旧工作流。

### Miner  `miner_hyperparams.json`

关键字段 皆为可选，默认值见文件：
- `miner_task_count`, `miner_task_seed`
- `validator_task_count`  可选，本地提交前验证套件大小
- `fec_cache_size`, `fec_train_examples`, `fec_valid_examples`, `fec_forget_every`
- `submission_cooldown_seconds`, `submit_only_if_improved`, `max_submission_attempts_per_generation`
- `validate_every_n_generations`
- `sota_cache_seconds`, `sota_failure_backoff_seconds`
- `persist_state`, `persist_every_n_generations`
- `gene_dump_every`

### Validator 评估  `validator_hyperparams.json`

关键字段 皆为可选，默认值见文件：
- `epochs`, `task_count`, `task_seed`
- `default_task_type`, `default_input_dim`
- `log_task_scores`  仅当日志级别为 `DEBUG` 时才输出每任务分数
- `tasks.<task_type>.n_samples` 与 `train_split`  用于 `cifar10_binary`、`mnist_binary`
- `tasks.scalar_linear.train_samples` 与 `val_samples`

### 覆盖配置文件路径  可选

- `BITSOTA_MINER_HYPERPARAMS_PATH` 或 `MINER_HYPERPARAMS_PATH`
- `BITSOTA_VALIDATOR_HYPERPARAMS_PATH` 或 `VALIDATOR_HYPERPARAMS_PATH`

## 运行时调参  环境变量覆盖

仍支持环境变量以兼容旧方式，并且会覆盖上面的 JSON 默认值。

### Miner

- `MAX_EVOLUTION_GENERATIONS`  默认 `15`：直接挖矿循环的代数上限。通常由 `evolution.max_generations` 设置。
- `MINER_TASK_COUNT`  默认 `32`：每个 genome 打分时使用的确定性子任务数量，仅 CIFAR 任务使用。
- `MINER_TASK_SEED`  默认 `0`：生成确定性 miner 任务套件的随机种子。
- `MINER_FEC_CACHE_SIZE`  默认 `100000`：FEC 的 LRU 缓存大小，`0` 禁用。
- `MINER_FEC_TRAIN_EXAMPLES`  默认 `32`：FEC probe 训练子集大小。
- `MINER_FEC_VALID_EXAMPLES`  默认 `32`：FEC probe 验证子集大小。
- `MINER_FEC_FORGET_EVERY`  默认 `0`：每插入 N 次清空 FEC 缓存，`0` 禁用。
- `MINER_SUBMISSION_COOLDOWN_SECONDS`  默认 `60`：两次向 relay 提交之间的最小间隔秒数。
- `MINER_SUBMIT_ONLY_IF_IMPROVED`  默认 false：若启用，仅在验证分数超过本地最佳时才提交。
- `MINER_MAX_SUBMISSION_ATTEMPTS_PER_GENERATION`  默认 `1` 或 `3`：每代的提交尝试次数。
- `MINER_VALIDATE_EVERY_N_GENERATIONS`  默认 `1`：限制本地类似 validator 的打分频率。
- `MINER_SOTA_CACHE_SECONDS`  默认 `30`：缓存 SOTA 获取结果的时长。
- `MINER_SOTA_FAILURE_BACKOFF_SECONDS`  默认 `5`：SOTA 获取失败后的退避秒数。
- `MINER_GENE_DUMP_EVERY`  默认 `1000`：调试 gene dump 的频率 代码中启用时生效。

### Validator 评估

- `VALIDATOR_TASK_COUNT`  默认 `128`：validator 评估套件中的确定性任务数量。
- `VALIDATOR_TASK_SEED`  默认 `1337`：validator 任务套件随机种子。
- `LOG_VALIDATOR_TASK_SCORES`  默认 `0`：若为真且 validator 日志为 `DEBUG`，输出每任务分数。

### CIFAR 任务缓存

- `CIFAR_TASK_CACHE_MAXSIZE`  默认 `512`：准备好的 CIFAR 投影任务的 LRU 缓存大小。

### EVM key  capacitor 模式

- `EVM_PRIVATE_KEY`  可选：当 `evm_key_path` 未提供且钱包 `h160/` 文件不存在时，由 `common/contract_manager.py` 使用。

### 脚本 key  运维工具

- `PRIVATE_KEY`  某些脚本必需：当未提供 `--keyfile` 时由 `scripts/common.py` 使用。

---

## 桌面 GUI 开发覆盖  `new_gui`

冻结版桌面构建会忽略覆盖配置并使用硬编码默认值。本地与开发运行支持通过 JSON 覆盖端点配置。

### `BITSOTA_GUI_CONFIG`

如果设置了 `BITSOTA_GUI_CONFIG`，它应指向一个 JSON 文件，用于覆盖 GUI 的端点配置：

- `relay_endpoint`  字符串
- `update_manifest_url`  字符串
- `pool_endpoint`  字符串
- `test_mode`  bool
- `test_invite_code`  字符串
- `miner_task_count`  int
- `validator_task_count`  int
- `miner_validate_every_n_generations`  int
- `problem_config_path`  字符串

如果未设置 `BITSOTA_GUI_CONFIG`，应用会依次查找：
- `./bitsota_gui_config.json`
- `./gui_config.json`
- `~/.bitsota/gui_config.json`
