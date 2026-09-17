# A 阶段：执行租约任期隔离

基线：`origin/main`（`becb865c2`）。

执行资格使用 `(task_id, runner_id, run_id, attempt_id)`，复用已有
`tasks.lease_attempt_id`，不修改数据库结构。同 runner、同 run 重新获取
lease 后，旧句柄不能继续续租、结算或写入受保护状态。renew 延长原任期，
预先获取的 lease 和 heartbeat 交接继续使用原对象。

## 写入边界

- 续租、释放、错误结算、恢复输入、clarification 和 interaction staging
  校验完整任期；缺少 acquisition token 视为失效。
- 心跳按完整任期分组，旧批次结果不会影响后继任期；同任期订阅共享心跳。
- WebSocket 实时输入使用数据库快照查找本事件循环中已注册的 lease，
  找不到持有者时沿用 defer 路径，不把任务行快照直接作为执行资格。
- 结果结算先用条件 UPDATE 取得写锁，再读取及提交，兼容 SQLite 和
  PostgreSQL；标题、Workforce 状态和 token 统计也带原任期。
- 普通 Trace 和 outbound 写入先取得任期条件锁。checkpoint 沿用条件
  指针 UPDATE，保留任务已删除时 required / best-effort 的错误分类。
- TTL 回收快照带 attempt，旧扫描不能清除同 runner、同 run 的新任期。

本实现直接适配 main 的现有持久化路径，不包含尚未合并的执行事件 writer。
mailbox、命令生命周期接管及获取入口统一属于后续阶段；命令回执继续使用
现有 `claimed_by + attempt_count` 条件保护，原有 acquisition 行为保留。

## 验证

`tests/web/services/test_task_lease_epoch.py` 在 SQLite 和 PostgreSQL 上
验证旧／缺失 token、心跳交接和延迟批次、并发重新获取、取消清理、结算锁、
过期扫描及各结果写入点。相关 API、互动、checkpoint 和统计用例一起回归。

移植后 32 个相关测试文件共 **1323 项通过**，包含 SQLite 和 PostgreSQL
用例。全部适用 pre-commit 检查及 `git diff --check` 通过。

## 部署边界

上线前排空或停止旧 worker 的执行、心跳和命令处理。旧进程仍能发送不含
attempt 条件的 SQL，因此不能依赖新代码约束混跑中的旧 worker。
本阶段保护内部持久状态，不撤回已发出的外部请求。
