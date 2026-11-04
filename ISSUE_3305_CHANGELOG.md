# Issue #3305 修改说明

Issue 链接：<https://github.com/xorbitsai/xagent/issues/3305>

## 1. 问题背景

MCP Connector 的配置加载存在两套逻辑：

- `WebToolConfig` 已经负责 team scope 解析、可见 MCP Server 查询和 actor stdio 配置加载。
- `mcp_connector_session.py` 又重复实现了 team scope 解析和 actor stdio 配置筛选。

这会让同一个 connector probe 重复执行相同的权限和可见性解析。带有 `connector_team_id` 时，一次 `run_connector_probe()` 原本可能触发三次 team scope hook：

```text
run_connector_probe
├── find_visible_server
│   └── resolve_team_connector_selection_or_raise  # 第一次
└── list_connector_tools
    ├── find_visible_server
    │   └── resolve_team_connector_selection_or_raise  # 第二次
    └── load_mcp_server_config
        └── _load_mcp_server_configs
            └── resolve_team_connector_selection_or_raise  # 第三次
```

actor stdio connector 还存在另一个问题：调用方会先解析完整的可见 server 集合和全部 actor stdio 配置，最后才按 `app_id` 找出一个配置。这会增加数据库读取、凭证处理和不必要的配置构建成本。

## 2. 修改目标

本次修改对应 Issue #3305 的目标：

1. 让 `WebToolConfig` 成为 connector 配置加载的统一入口。
2. 允许调用方把已经解析的 team selection 传给后续加载流程。
3. 增加按 `app_id` 获取 actor stdio 配置的入口。
4. 保留 probe 原有的 catalog identity 预检查和结果行为。

## 3. 修改内容

### 3.1 `WebToolConfig` 增加 team scope 统一入口

文件：`src/xagent/web/tools/config.py`

新增：

```python
resolve_connector_team_selection()
visible_mcp_server_query(team_selection=...)
```

`visible_mcp_server_query()` 负责复用已解析的 team selection；没有传入时，才自行解析。这保留了原有调用方的兼容行为。

### 3.2 MCP server 单配置加载支持复用 team selection

修改：

```python
load_mcp_server_config(server_id, team_selection=...)
_load_mcp_server_configs(only_server_id=..., team_selection=...)
```

当调用方已经完成 team scope 解析时，`WebToolConfig` 不再重复调用 team hook。

### 3.3 增加 actor stdio 单配置入口

新增：

```python
load_actor_stdio_config(app_id, team_selection=...)
```

`list_actor_stdio_connector_tools()` 现在通过 `WebToolConfig` 加载 actor stdio 配置，不再直接调用：

```python
resolve_actor_mcp_stdio_configs
```

配置的最终选择仍然按照 `app_id` 完成，原有 shadowing、policy 和 execution scope 行为保持不变。

### 3.4 `mcp_connector_session.py` 复用统一入口

修改后的流程：

```text
list_connector_tools
├── 创建 WebToolConfig
├── 解析 team selection（如果调用方没有传入）
├── 使用同一个 selection 查询可见 server
└── 使用同一个 selection 加载 server config
```

`find_visible_server()` 和 `_visible_server_query()` 现在可以接收外部传入的 `team_selection`，避免重复解析。

### 3.5 `connector_probe.py` 只解析一次 team scope

`run_connector_probe()` 的内部流程现在是：

```text
1. 通过 WebToolConfig 解析一次 team selection
2. 使用该 selection 查找 server
3. 执行原有 catalog identity 检查
4. 将同一个 selection 传给 connector session
5. 加载 connector config 并执行 probe
```

这样保留了 `_owns_catalog_identity()` 前置检查，同时避免后续 connector session 再次解析 team scope。

## 4. 修改前后对比

| 内容 | 修改前 | 修改后 |
| --- | --- | --- |
| team scope 解析 | `connector_probe`、`mcp_connector_session`、`WebToolConfig` 各自触发 | 由 `WebToolConfig` 统一解析，并可复用结果 |
| 一次 server probe 的 team hook | 最多三次 | 一次 |
| 可见 server 查询 | session 层手工组合 | `WebToolConfig.visible_mcp_server_query()` |
| actor stdio 加载 | 解析全部配置后按 `app_id` 筛选 | 通过 `WebToolConfig.load_actor_stdio_config()` 统一处理 |
| catalog identity 预检查 | 保留 | 保留 |
| 原有 connector 结果状态 | 原有行为 | 保持不变 |

## 5. Issue 验收标准对应情况

### 验收标准一

> connector session 不再直接调用 `resolve_team_connector_selection_or_raise` 或 `resolve_actor_mcp_stdio_configs`。

已完成。两个底层 resolver 已从 `mcp_connector_session.py` 的执行逻辑中移除，统一由 `WebToolConfig` 调用。

### 验收标准二

> 带有 `connector_team_id` 时，一次 `run_connector_probe` 只调用一次 team hook。

已完成调用链改造：`connector_probe` 先解析一次 selection，再传给可见性查询和配置加载。

### 验收标准三

> 相关 actor policy、connector session 和 probe 测试保持行为不变。

代码已通过 Python 编译检查：

```bash
python3 -m compileall -q src/xagent/web/services/mcp_connector_session.py \
  src/xagent/web/services/connector_probe.py \
  src/xagent/web/tools/config.py
```

相关 pytest 已验证通过：

```bash
uv run pytest -q \
  tests/web/services/test_connector_session_actor_policy.py \
  tests/web/services/test_connector_probe.py
```

结果：

```text
74 passed
```


## 6. 重点查看文件

- `src/xagent/web/tools/config.py`
- `src/xagent/web/services/mcp_connector_session.py`
- `src/xagent/web/services/connector_probe.py`
- `tests/web/services/test_connector_session_actor_policy.py`
- `tests/web/services/test_connector_probe.py`

## 7. 介绍

概括本次修改：

> 原实现存在两个 MCP 配置加载入口，connector session 自己重复处理 team scope 和 actor stdio 配置，导致一次 probe 可能重复调用 team hook，并且为了一个 actor stdio connector 构建全部配置。本次修改把这些逻辑统一收口到 `WebToolConfig`，由 probe 解析一次 team selection 后向下传递，同时增加单个 actor stdio 配置入口，在保持可见性、权限和 probe 行为不变的前提下降低重复解析。

