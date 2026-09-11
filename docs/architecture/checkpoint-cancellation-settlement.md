# 取消与收尾完整性

## 写入完成后传播取消

`DatabaseTraceHandler._save_to_database` 和 `make_agent_outbound_handler` 的事件写入通过现有 `run_db_io_cancellation_safe` 执行。调用方取消后，等待已经启动的 worker 完成提交或回滚并关闭 Session，再传播 `CancelledError`。重复取消不会提前结束等待；worker 抛出的错误保留在取消异常的原因链中。普通 trace、required checkpoint 和 outbound 的原有错误分类继续由各自的 handler 负责。

正常外部取消的顺序为：取消请求 → 数据库 worker 完成并关闭 Session → 执行协程退出 → 停止心跳 → 收尾及释放任期。任期丢失和心跳故障继续使用现有处理策略。

取消不要求撤销已提交数据。已提交的 checkpoint 或 outbound 消息保留，但取消后的调用方不会继续执行正常成功广播。等待范围包括 worker 内的 checkpoint 清理，因此数据库操作缓慢也会延迟取消完成。本改动不提供立即停止数据库操作的保证。

## PostgreSQL 收尾直接取得强锁

checkpoint 插入并 flush 时，外键检查在 Task 行持有 `KEY SHARE` 锁，随后需要更新该 Task 的 checkpoint 指针。如果收尾先以无变化 UPDATE 取得写锁，再升级为 `FOR UPDATE`，两者可能互相等待。

`lock_task_lease_for_settlement_no_commit` 在 PostgreSQL 上直接执行带 task、runner、run、attempt 条件的 `SELECT ... FOR UPDATE`。checkpoint 已持有外键锁时，收尾等待期间不先持有阻止指针更新的弱写锁，checkpoint 可以完成后让收尾继续。SQLite 不支持该行锁，继续使用原条件 UPDATE。

以下四个有任期入口使用这一收尾锁：

- `task_orchestrator.finish_turn`
- `websocket._finalize_task_execution_result_isolated`
- `websocket._finalize_resumed_task`
- `workforce_runtime._sync_workforce_run_status_for_task_id`

后续状态读取、结果写入及任期释放仍在原事务中完成。无任期路径和普通 trace 使用的任期锁保持现有行为。

不能直接删除原有强锁：`FOR UPDATE` 还阻止引用 Task 的关联记录插入。受控验证中，删除它会使两个 WebSocket 收尾入口看到并发新增的文件元数据，从而改变文件链接解析结果。直接取得强锁保留这一保护范围。

## 回归覆盖

- `test_task_orchestrator.py`：通过真实 runner、持久化 handler 和数据库 Session 验证 checkpoint、普通 trace、outbound 的成功及失败路径。两次取消期间 worker 未结束、心跳未停止、收尾未开始；放行后检查 Session 关闭、失败回滚、任务收尾及 delivery 状态。
- `test_task_settlement_locking.py`：SQLite 和 PostgreSQL 上四个入口拒绝旧任期；PostgreSQL 上四个入口持锁时继续阻止 Task 更新、删除和关联记录插入；实际 checkpoint 与 `finish_turn` 的交错无锁升级死锁，并保留已提交的 checkpoint 指针。交错通过数据库锁等待状态确认。
- `test_db_runtime.py`：复用公共取消边界已有的重复取消和错误因果链测试。

这些是受控数据库和生命周期回归，不代表生产触发频率或所有部署配置。本改动不涉及心跳调度、TTL、连接池参数、工具统计、表结构或迁移。
