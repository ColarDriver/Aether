# 07 配置建议：Hermes 风格 vs .env 最小化

## Hermes 的配置方式（参考）

Hermes 是“双层配置”：

1. `~/.hermes/config.yaml`
   - 模型、provider、fallback、toolsets、运行策略
2. `~/.hermes/.env`
   - API key、token、部分 provider/base_url

并通过统一 loader 在启动时加载，支持 profile 隔离与优先级控制。

## Aether 当前建议

短期建议采用“分阶段”策略：

### 阶段1（当前开发）

- 允许 `.env` 直连（最低成本，便于快速测试通过）
- 保持 provider 构造参数可显式注入（测试更稳定）

### 阶段2（稳定后）

- 引入 `aether/config` 的统一配置入口（类似 Hermes 的 config.yaml + .env 合并）
- 统一处理：
  - model/provider/base_url/api_key
  - 环境优先级
  - profile/多环境隔离

## 结论

- 如果目标是“先验证功能链路”，`.env` 足够
- 如果目标是“长期可维护与多环境一致性”，应逐步收敛到 Hermes 式统一配置层
