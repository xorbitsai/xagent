# Phase B：任务统一调度与执行协调

状态：设计提案，已实现未接线的租约与协调器生命周期基础。日期：2026-09-08。
代码基线：`origin/main`，`63f178b08`（已合并 Phase A，#2188）。
本文取代旧分阶段设计中关于 B 的部署、共存和所有权生命周期假设。

## 当前实现进度

实现 worktree：`phase-b-task-coordination`，分支：`feat/task-execution-coordination`。
实现基线：`origin/main` 的 `5b233592c`；上述 `63f178b08` 是设计时基线。

第一批实现位于 `task_coordinator_service.py`：

- 一份三元任务租约与携带固定 run 的执行上下文，执行上下文没有独立续租／释放接口。
- 获取、续租和释放参与调用者事务；获取不修改业务状态，也不允许同进程重入。
- 过期但尚未分类的任期不能被直接覆盖；原持有者仍可在接手前续租。
- 按完整控制状态快照开始新 run／恢复原 run，保留 owner 任期；新 run 清理指针，恢复保留指针。
- 任务和执行条件写锁；旧 run 或旧任期的执行上下文不能写入后继执行。
- 释放保留任务业务状态；RUNNING 必须先结算，调用者还需完成后台清理和队列检查。

补充实现位于 `task_coordinator_runtime.py`：

- 每个 worker 事件循环共用一个 `TaskCoordinatorRegistry`，并发 ensure 共用一次获取，
  重复唤醒复用实例和心跳；一个等待者取消不影响其他等待者。
- 没有调用者取得实例时，获取事务即使晚提交也会等待完成并清理；启动失败、关闭和
  重复取消都等待资源退出后才移除注册项。
- 每个协调器只有一个任务租约心跳，执行结算与后台清理期间继续续租。
  失去所有权会取消并 drain 执行；连接池超时按正常心跳间隔重试，不误判接手。
- `submit_execution(admit, execute, settle)` 在让出事件循环前占住唯一运行句柄；
  准入和结算分别在任务／执行条件写锁事务内完成，业务回调不自行管理租约。
- 准入被拒绝时回滚该事务；取消撞上准入提交时先取得提交结果，再结算已经开始的 run。
  数据库提交结果不确定时停止当前协调器，保留任期和执行证据供恢复使用。
- 非 RUNNING 过期任务负责人可以原子回收，保留业务状态和 checkpoint；与旧任期续租
  竞争时只有一方成功。RUNNING 仍必须经过原有 checkpoint 恢复分类。

本批未接入 dispatcher、命令协调循环和现有执行入口。当前生产路径仍使用 Phase A 的接口；
新模块不是第二套运行中的租约服务，也没有添加类型兼容别名。
后续需要将各入口迁入上述准入边界，完成命令效果凭据、恢复及队列检查后再整体接线。
`close()` 是显式关闭／故障退出，不冒充已经实现了释放与命令入队之间的空闲检查。
这批代码不能单独视为 Phase B 已完成或已经可以切换。

初批验证：租约原语 54 项 SQLite/PostgreSQL 用例；连同 A 阶段回归共 244 项通过。
结算取消修复前验证：两个新模块共 112 项 SQLite/PostgreSQL 用例；连同 A 阶段租约、
managed lease 和恢复回归共 302 项通过。包含真实进程终止、续租／回收竞争、启动取消、
共享等待者、运行句柄互斥、旧任务负责人回调、关闭 drain、连接池超时及提交结果不确定。
全部适用 pre-commit 检查（包括完整 mypy）通过；本批未运行完整 Web CI。

结算取消修复后，两个新模块共 116 项 SQLite/PostgreSQL 用例通过：新增关闭撞上成功结算、
重复取消后立即接手的回归；提交结果确实未知时仍保留租约供恢复处理。

关闭期间唤醒修复后，共 122 项双数据库用例通过。`ensure()` 等待旧实例清理完成后重试，
不覆盖仍在退出的实例；多个等待者共用后继实例，取消等待不取消清理，全局关闭不重建实例。

### 执行入口迁移核对

共同边界已经可独立运行和测试，但下面的生产包装器尚未迁移，不能直接当作 `execute`
回调传入：它们仍会 acquire、启动心跳或 release。正式接线必须移出这些生命周期动作，
保留准入事务中的业务校验和结算事务中的投影；不得用四元／三元类型转换冒充迁移。

| 入口与现有边界 | 迁入共同边界时必须保留的工作 |
| --- | --- |
| WebSocket 新轮次：`_claim_turn_no_commit`、`_schedule_bg`、`finish_turn` | turn 身份、文件绑定、用户校验和持久消息；执行与异步清理归运行句柄，结算不 release |
| WebSocket 恢复：`_acquire_resume_task_lease`、`execute_resume_background` | owner/run 校验、原状态与 checkpoint；初始化失败恢复原状态，保留 interaction marker 清理 |
| A2A：`_acquire_a2a_resume_prelease_sync`、`_resume_input_required_a2a_task` | agent/source、控制状态与旧 run 校验；输入和恢复安排必须仍可追踪 |
| v1 回答：`_acquire_reply_prelease_sync`、`_schedule_waiting_reply_resume` | 问题级 CAS、目标 run/version 与回答校验；恢复失败保留等待语义 |
| 渠道：`_prepare_channel_task_sync`、`ManagedTaskLease` | 渠道 actor/agent 校验、回答恢复与文件身份；新建任务事务需拆出可持久交接，不能把未提交 ORM 行交给 registry |
| 公共执行：`chat.py` 的 `manage_task_lease` 分支 | quota/scope、tracker 清理和 usage 写入必须在运行句柄 drain 完成前结束 |
| Workforce：`_claim_turn_no_commit`、`sync_workforce_run_status` | archived/config 校验和 Task/WorkforceRun 同事务投影，结算不单独释放 |

这份核对明确了接线改动位置，不代表现有入口已经共用新协调器。第 13 节第一步中的
“所有入口 admission”只有在这些调用点迁移并验证后才能勾选；本批只验收共同边界及其生命周期。

## 1. 目标与部署前提

Phase A 已保证失效执行任期不能续租或提交受保护状态。Phase B 在此基础上，
将任务命令消费与执行调度交给同一个任务负责人，取消每条命令独立选 worker 的过程。

部署采用一次停机切换：停止全部旧 worker，再启动新版本。没有新旧版本混跑，
不设计任务版本归属、灰度分流、两套 dispatcher 并行消费或运行中热迁移。
数据库中仍可能残留旧命令和未决执行，因此启动恢复不能省略。

完成 B 后必须满足：

1. 一个任务同时只有一个有效任务负责人任期；同进程重复唤醒不重新 acquire。
2. 命令处理、Runner 调度和结果写入共用该任期；不另选一个命令负责人。
3. Runner 执行期间，任务协调器仍可处理持久命令队列与执行完成事件。
4. 接收、安排执行、输入已应用、命令完成和任务完成具有不同的持久语义。
5. 崩溃后根据持久证据恢复；不以命令超时或进程更换作为安全重发的证明。
6. 对外消息身份、授权、run 约束、文件处理、错误分类与控制顺序继续成立。

不承诺：任意外部副作用恰好一次、不可中断工具立即停止、透明恢复所有过期执行。

## 2. 范围与 B/C 边界

B 的业务主切片是第一方 WebSocket 持久 MESSAGE，以及同任务的 PAUSE、RESUME、
CANCEL、新轮次和等待回答恢复。不能只迁 MESSAGE 而让控制命令保留独立 claim。

新旧代码不混跑，也不能让新版本内部的第一方旁路重新争抢同一任务：

- B 必须提供所有第一方执行入口共用的任务租约获取、注册和退出边界。
- A2A、v1、渠道和 Workforce 的现有业务处理可以保留，但启动执行要通过该边界，
  将原有执行作为任务负责人管理的运行句柄；不得创建第二个心跳所有者或自行释放任务租约。
- 已有任务负责人时，入口只能向它递交工作或沿用已有 defer/reject 行为；不能从任务行
  抄 token，也不能因为 runner_id 相同而重入 acquisition。
- 远端任务负责人通过持久命令和扫描发现工作；本进程不能直接取得远端 Runner。
- C 再把这些入口的全部接收、回执与业务编排统一到持久命令队列协议。

这是对早期“B 完全不碰其他入口”的修正：B 需要最小所有权接入，C 才是完整业务迁移。
若某入口无法做到该接入，B 不满足统一所有权的上线条件，不能悄悄保留绕行路径。

外部 executor 的注册接口和授权／scope 契约保留。它对本任务的命令分派仍须经过
任务负责人；对其外部系统的交付保证保持原契约，不冒充第一方 checkpoint 证明。
无法证明安全的外部重复执行继续报告不确定结果，不能用新任务负责人自动重试。

## 3. 当前代码中需要改变的事实

| 当前实现 | B 的要求 |
| --- | --- |
| acquire_task_lease_no_commit 同时设置 RUNNING、run、control_state | 拆开取得任务租约与进入执行状态 |
| 同 runner 可再次 acquire，生成新 attempt | 有效任期不能重入；同进程复用真实协调器实例 |
| heartbeat 仅为 RUNNING 延长，其他状态返回 SETTLEMENT_READY | 任务负责人心跳覆盖命令处理和收尾；执行状态不直接决定任务负责人是否终止 |
| dispatcher claim 命令并启动独立心跳 | dispatcher 发现任务工作，唤醒／竞争任务负责人 |
| 命令回执按 claimed_by + attempt_count 提交 | 任务租约条件校验 + 命令状态／处理次数 CAS |
| WebSocket handler 内部预约、claim delivery、启动后台执行 | 任务负责人安排操作，handler 保留业务校验与准备 |
| finish_turn、渠道结算会释放 task lease | 执行结算与任务负责人释放分开 |
| 部分 resend_safe 依赖 attempt_count == defer_count + 1 | 第一方新路径改查持久交付证据，不能沿用计数推导 |

## 4. 一份任务租约与执行上下文

**只有一套可获取、续租和释放的任务租约。执行上下文不是第二份租约。**
任务负责人是持有租约的逻辑身份；任务协调器是管理该租约和任务运行的实例。

### 4.1 TaskLease（唯一的任务租约）

任务所有权身份为 `(task_id, runner_id, attempt_id)`，使用现有 tasks 的
runner_id、lease_attempt_id、lease_expires_at、last_heartbeat_at 字段。
不新增第二套所有权 token，不新增另一条可独立过期的命令租约。

取得任务租约只改变所有权字段，不修改 Task.status、control_state、run_id、checkpoint。
因此取消一个 PENDING 任务不必先把它标为 RUNNING，也不需要虚构一次 run。
只有任务协调器负责获取、续租和释放这份租约。

### 4.2 TaskExecutionContext（执行上下文）

执行上下文携带任务租约和固定的 run_id，由真实任务负责人在合法 run 事务完成后创建，
不能从数据库观察值创建。它不单独 acquire、续租或释放，也没有独立到期时间。

执行 writer 从上下文取得 `(task_id, runner_id, run_id, attempt_id)`，继续使用 A 的
完整 run + attempt 条件写入。缺少 run 或 attempt 不能执行。

任务负责人可以先处理不启动执行的命令；转入执行时，在同一份任务租约上创建执行上下文。
切换到新 run 时，只能在旧执行及其回调 drain 完成后，在受保护事务中更新 run，并创建
新的 TaskExecutionContext。旧上下文仍固定在旧 run，不会随新 run 自动更新，仍被条件拒绝。

命名迁移说明：当前 Phase A 代码中的 TaskLease 是包含 run_id 的四元执行句柄。
本文的 TaskLease / TaskExecutionContext 是 Phase B 的目标命名；实施时需显式迁移类型和
调用点，不能直接把现有四元校验删成三元校验。本次文档改名不修改现有代码类型。

### 4.3 心跳与租约函数

- 任务协调器只启动一套任务租约心跳，按任务、进程和 attempt 管理。
- 执行上下文观察同一份租约的有效性，不额外开执行续租循环。
- A 的执行条件校验继续严格匹配 run；不得将所有执行 writer 放宽为三元身份。
- 任务租约续租匹配完整所有权身份，不因为 PAUSED／WAITING／COMPLETED 立即停止。
- 执行收尾信号与租约丢失信号分开，不复用 SETTLEMENT_READY 表示任务协调器已退出。
- 接口迁移须覆盖现有 managed lease 和 prelease 交接；旧 helper 只在调用点迁完后移除。

这是 B 对 A 的生命周期扩展，不是放松 A 的执行写入隔离。

## 5. 任务租约的取得、协调与退出

### 5.1 唤醒与取得

本进程维护 `task_id -> coordinator` 注册表。注册过程需要单次创建保障，
且注册表本身不赋予数据库所有权。失败的创建、退出和取消必须移除对应占位。

1. 若已有本进程协调器实例，发送唤醒，不重新 acquire。
2. 否则通过短事务尝试取得任务租约；CAS 允许无任务负责人或已完成安全过期处理的任务。
3. 不允许仅凭同 runner 或非 RUNNING 状态覆盖有效任务负责人。
4. CAS 失败表示其他任务负责人负责；保持持久工作，由轮询补偿通知。
5. CAS 成功后注册实例并启动心跳；启动失败按原 attempt 条件清理。

TTL 后尚未被接手时允许原任务负责人尝试 renew；renew 和接手竞争由数据库 CAS 决定。
接手后旧 attempt 一定失效。所有进程遵守同一判断，时钟来源和 TTL 配置沿用现有策略。

### 5.2 本地状态机

| 协调状态 | 行为 |
| --- | --- |
| ACQUIRING | 单次取得数据库任务所有权，尚不启动 Runner |
| ACTIVE | 处理持久命令队列／完成事件，可以有一个在运行的 Runner |
| QUIESCING | 停止安排新执行，完成必要回执和后台清理 |
| RELEASING | 检查无运行句柄及可立即处理工作，条件释放 |
| LOST | 不再安排或提交工作，取消并 drain 原任期工作 |
| CLOSED | 停止心跳，移除自己的注册项 |

这些是运行时状态，不是新增 Task.status 枚举。Task.status 继续表达用户任务状态。

### 5.3 Runner 与持久命令队列

协调循环等待持久命令队列唤醒、Runner 完成、所有权失效和关闭事件。
不能从启动到完成一直 await Runner 而不处理命令。

同任务只安排一个活动 Runner；普通工具内部并行能力不变。
命令应用及上下文变更复用 Runner 现有安全注入／暂停边界，不能直接并发修改 context。
准备文件等命令操作可以异步等待，心跳和所有权失效处理必须继续运行。

### 5.4 释放与入队竞态

释放前必须已结算 Runner、必要命令结果和所有会写状态的回调。
在任务条件写锁保护下复查可立即处理的工作；有工作则继续，无工作则清除任务负责人。

命令入队事务也需与任务负责人的发现／释放建立一致锁顺序：先任务，再命令。
若入队先提交，释放检查看到工作；若释放先提交，入队后的通知或持久扫描重新唤醒。
通知只是加速；扫描必须覆盖无任务负责人、可处理命令和到期重试，避免永久滞留。
未来重试尚未到期时可以释放，不为等待 backoff 长期持有空闲任务负责人。

## 6. 持久命令队列与命令状态

复用 TaskExecutionCommand，保留唯一消息身份、payload、actor/scope、target_run_id、
target_state_version、状态、结果和终态事件。

状态继续使用 pending / processing / succeeded / failed 等现有常量，
实际终态名称以现有代码为准，不改变客户端枚举。

- pending -> processing：当前任务负责人在 task 条件写锁下选取最早合法命令；
  增加 attempt_count，并以命令状态 CAS 标记处理，不产生独立命令所有权。
- processing -> terminal：任务租约条件校验、命令 id、状态和本次 attempt_count 一起校验。
- processing -> pending：在明确可重试时设置重试时间和原有 failure/defer 计数。
- 失去任务负责人：旧回调不能 finish/fail/defer；后继任务负责人先核对回执再恢复 processing。

attempt_count 保留为处理次数及同任务负责人异步回调的 CAS，不代表独立选举。
claimed_by 与 claim_expires_at 不再参与新路径授权，不续租；暂保留数据库列供旧状态审计。

### 6.1 最小 schema 提案

在 task_execution_commands 增加两列，放在一个 migration 中：

| 字段 | 用途 |
| --- | --- |
| available_at，可空时间 | pending 命令的下次处理时间，替代对 claim TTL 的混用 |
| effect_receipt，可空 JSON | 私有的持久效果凭据，和业务结果 result 分开 |

不新增消息队列表，不新增任务负责人字段，不新增新旧版本归属字段。
首版不猜测添加索引；复用当前任务顺序索引，根据实际查询计划决定是否需要迁移内调整。

effect_receipt 使用显式版本和受控结构，至少记录 effect kind、run、命令／turn 身份，
必要时记录 checkpoint 标识、控制目标或持久启动信息。它不是给客户端直接返回的 result。
凭据是已发生效果的证明，不因 checkpoint 修剪、worker 更换或失败重试而随意清空。
只记录必要关联数据，不重复存储敏感 prompt 或文件内容。

旧记录没有 receipt，必须走第 10 节的旧状态恢复规则，不能默认未执行。

## 7. MESSAGE 的接收、安排、应用与回执

### 7.1 接收

授权、命令 id、turn_id、payload 冲突及文件归属检查沿用当前契约。
提交命令／接收记录后才确认已接收。接收不会 acquire 任务负责人，也不会承诺已执行。
命令消费时仍校验需要动态验证的授权和历史 actor/scope，不能用任务负责人绕过它们。

### 7.2 在已有 Runner 中注入

任务负责人按原任务/run 规则准备输入，通过 Runner.post_user_message 等现有接口注入。
继续使用 turn_id、POSTED_FRESH／POSTED_REPLAY、checkpoint 和历史投影机制。

输入 checkpoint、任务 checkpoint 指针和该命令的 input_checkpointed receipt，
必须在同一个受任务负责人 + run fence 保护的数据库事务提交。
需要给 checkpoint writer 传递明确的命令关联，不能仅靠协程 ambient command 推断。
建议将待确认命令关联放入该 checkpoint 的内部元数据，由 writer 验证 task/run/turn 后处理。
普通 checkpoint 不因此扫描或重新确认所有历史命令。
这里的“已应用”指输入已进入可恢复执行上下文，不代表模型已经使用它完成推理。

writer 同时校验命令处于本次合法应用步骤、身份/payload 关联一致；错误关联必须回滚。
receipt 中保留可追踪的 checkpoint 身份，但不能用级联删除让 checkpoint 修剪抹掉凭据。

一旦 input_checkpointed 已提交，该命令的输入效果成立；之后广播失败不能改成安全重发。
POSTED_REPLAY 只说明 Runner 找到了相同 turn，不能单独替代数据库 receipt：
恢复时仍须证明对应持久 checkpoint，不能接受未成功落盘的内存 context 作为完成证据。
必需 checkpoint 失败后，需要撤销未确认的 context 变更，或使该 context 失效并从
已确认 checkpoint 重建；不能让下一次调用从内存找到相同 turn 就误报成功。

### 7.3 无 Runner，需要启动／恢复

准备工作完成后，由任务负责人在事务中完成合法 run 状态转换，并写入 input_scheduled
凭据及可恢复的输入关联；然后调度 Runner。命令不能凭本地 create_task 成功就完成。

允许沿用既有“持久 handoff 后命令完成”的外部语义，但 input_scheduled 必须能独立
恢复：原始输入、目标 run、必要文件身份和执行配置必须来自现有持久记录，不能只留在内存。

在执行任何模型／工具副作用前，Runner 必须提交首个完整、可恢复 checkpoint，并将
receipt 从 scheduled 推进为 checkpointed。不能生成缺少运行状态的伪 checkpoint。
已有恢复流程读取到 checkpoint 后，遵循其工具／执行不确定状态，不因命令已 scheduled
就从头启动。待启动工作发现还必须扫描 scheduled receipt，即使命令本身已终态。

若现有启动 API 无法提供该边界，B 必须补上启动边界后才能采用 scheduled 完成语义。
不能以纯内存预约作为替代；在该能力完成前也不能开启新路径。

启动安排的发现条件必须同时匹配原 run、合法任务／控制状态和未完成的启动效果。
若启动前已取消或初始化已终止，相关事务须将启动凭据标为不再可执行；扫描不能仅
凭 scheduled 字样重新启动已取消／完成的任务。后续 MESSAGE 在初始 context 就绪前
不得假装注入成功；控制命令仍可按上述持久状态完成取消或暂停请求。

### 7.4 回执恢复

| 证据 | 行为 |
| --- | --- |
| checkpointed | 不重复注入；按持久结果补命令回执／投影 |
| scheduled，尚无初始 checkpoint | 新协议保证副作用未开始；根据持久启动信息继续初始化 |
| scheduled，已有 checkpoint | 对齐 checkpoint 与 receipt，按原恢复规则继续，不从头重跑 |
| 新协议命令无效果凭据 | 核对没有成功业务事务，再按原身份重试 |
| 旧协议或外部执行证据不足 | 保守处理为不确定，不自动断言未应用 |

数据库 commit 回执丢失时必须重新读取持久证据；读取也失败则保留未决状态。
不得立即写入 failed，也不得立即进行第二次注入。

## 8. PAUSE、RESUME、CANCEL 与顺序

默认保持现有同任务命令顺序，不增加控制优先队列或 CANCEL 越过前序命令的规则。
长时间 Runner 执行不能占住 MESSAGE 的命令队头；文件准备或退避仍可能阻塞后续命令，
这是保留的顺序契约，不能宣称 B 提供任意时刻的立即取消。

- PAUSE：校验目标 run/version；持久记录合法 pause_requested 和 control receipt，
  再向 Runner 发信号。命令成功表示请求被接受，任务实际暂停由后续状态通知表达。
- CANCEL：同样先持久化合法控制效果，再请求取消。重复请求按已有幂等规则处理，
  旧 run 的取消不能作用于新 run。外部已发请求不承诺撤回。
- RESUME：有同 run 活动执行时沿用 already-in-progress；等待暂停收尾时沿用 defer；
  允许恢复时写入持久恢复安排，由同任务负责人启动，不能再次 acquire。
- 无活动 Runner：直接按状态机处理，不为执行控制命令虚构一次 RUNNING 转换。

控制状态变化与 receipt 必须同事务。信号发出前崩溃时，新任务负责人根据持久 control_state
继续完成控制动作；不能把信号是否已经发送作为唯一证据。
所有命令结果、终态事件、错误文本和个人／任务广播的权限边界沿用现有适配器。

## 9. 结算、等待与新轮次

执行 finalizer 在原 run + attempt 下提交结果、文件、历史和任务状态，不自行清除任务负责人。
任务协调器在命令结果和必要后台清理结束后决定继续处理工作或释放。

- WAITING_FOR_USER：确认等待 checkpoint、互动状态和相关回执已提交，可以释放。
- 新答案：新任务负责人恢复原 run，保留原 checkpoint 语义。
- COMPLETED 后新轮次：旧执行 drain 后创建新 run，按现有规则清理指针及结果字段。
- 同一问题的两个答案：保留 interaction CAS；一个任务负责人不替代回答唯一性约束。
- 输入投递的独立资格竞争只在受任务负责人完整覆盖的路径撤掉；消息去重和应用状态检查保留。

## 10. 崩溃、过期与旧状态接管

### 10.1 新版本运行中故障

任务负责人丢失后停止新操作并 drain 本地运行；A 的执行 fence 继续保护延迟返回。
新任务负责人不直接覆盖一个仍持有有效 lease 的任务。

过期 RUNNING 任务继续使用现有 PAUSED／FAILED 分类和恢复依据，不一律透明重启。
过期的非 RUNNING 任务负责人需要可回收，但不能把任务状态改成 RUNNING 或清除有效 checkpoint。
恢复器、task_has_live_runner 等观察函数必须区分“存在任务负责人”和“存在活动执行”。
判断启动是否已经发生时使用持久 checkpoint／scheduled 协议，不能只看内存注册表。

### 10.2 一次停机部署

1. 进入维护状态，停止接收新的执行工作和领取命令。
2. 排空交付事务、回执和正在完成的持久写入；对长执行请求安全暂停，无法排空则记录未决。
3. 停止所有旧 Runner、命令心跳和 task 心跳，确认旧进程不会再次写数据库。
4. 执行 schema migration，并在业务入口／dispatcher 开启之前运行旧状态整理。
5. 对旧 pending 保留顺序、payload 和重试时间；旧 processing 不因 TTL 清除而直接重跑。
6. 旧 processing 有匹配的持久 checkpoint／交付证据时补结果；明确未应用时恢复 pending；
   其余保留不确定并沿用失败／人工重试处置，不能伪造已应用凭据。
7. 清理旧 claim 字段和失效任务负责人必须与分类结果绑定，操作可重入并保留计数审计。
8. 启动新任务负责人发现循环，完成恢复检查后开放接收入口。

旧状态整理可并行启动多个新 worker，但每个任务的整理必须受数据库 CAS/锁保护。
干净停机可以减少未决记录，不能作为没有未决记录的证明。

### 10.3 回滚

不支持让旧版本直接读取并重放新 processing/scheduled 命令。
回滚需要再次停机，核对并转换新协议未决状态；不能只回滚代码或自动删除新增字段。
优先采用修复新版本后恢复运行，避免在证据不完整时跨协议重放。

## 11. 事务、锁和异常处理约束

- 统一任务 -> 命令 -> checkpoint／interaction／输出关联的锁顺序；迁移前核对现有
  路径中的实际锁序，禁止另一入口反向取锁。
- PostgreSQL 使用条件写锁；SQLite 在读后写事务之前取得条件 UPDATE 写锁。
- 数据库事务中不等待 LLM、工具、文件上传或网络广播。
- 任务租约条件校验 未命中是失去资格；数据库超时不是确认失去资格，不引入忙重试。
- 同一任务负责人下旧的异步命令回调也必须带本次命令处理次数，不能修改后续尝试。
- context 污染、失败 checkpoint 和未知 commit 必须保持保守恢复，不能仅靠内存去重。
- 通知失败不撤销已经提交的业务效果；个人回执身份不从任务成员关系推断。

## 12. 模块边界与拟议接口

以下是职责草案，不要求逐字实现接口名称，也不引入通用调度框架。

| 模块 | 改动 |
| --- | --- |
| 新 task_coordinator_service.py | 任务租约获取／续租／释放、注册表、协调循环、执行上下文派生 |
| task_lease_service.py | 保留 A 执行 fence，接入任务负责人生命周期，取消同 runner 重入入口 |
| task_command_transport.py | 任务工作发现、顺序读取、任务租约保护下的命令处理、结果与重试 |
| websocket.py | 保留协议适配与准备，执行／注入／控制交给任务协调器，移除目标路径重复资格竞争 |
| task_orchestrator.py、managed_task_lease.py | 执行结算不直接释放任务租约；统一运行句柄交接 |
| chat_history_service.py、trace_handlers.py、trace_event_staging.py | 效果凭据与 checkpoint／交付状态同事务 |
| runner.py / checkpoint 边界 | 明确命令关联、初始 checkpoint 和持久注入确认，保持工具逻辑 |
| task_command.py + 一个 migration | available_at、effect_receipt |
| app.py | 启动旧状态整理、任务负责人发现循环，以及有序关闭 |
| A2A/v1/渠道/Workforce 调用边界 | 最小任务租约获取／执行接入，完整业务迁移留给 C |

主要操作：ensure_coordinator(task_id)、wake(task_id)、submit_execution(coordinator, spec)、
execute_command(coordinator, command)、settle_execution(execution_context, result)、
release_when_idle(coordinator)。跨进程只传持久命令，不传 Python 运行对象。

## 13. 分步实施与上线门槛

不做新旧任务灰度，但实现可以分提交评审。未完成的组合不得进入实际部署。

1. **所有权基础**：任务租约与执行上下文分离，生命周期、非运行状态、所有入口 admission，
   取消同 runner 重入。单独验证，不立即替换线上消费者。
2. **持久效果与命令处理**：migration、effect receipt、checkpoint 原子边界、
   scheduled 启动协议、控制效果与回执恢复。
3. **协调接线与切换**：接入 WebSocket、控制与后台执行，停用旧命令 claim 心跳，
   实现启动旧状态整理和关闭流程，按停机方式一次启用。

三个步骤可以作为开发分支上的三个可审阅提交组；若拆 PR，前两步必须保持未接线，
正式消费者只在第三步整体切换。不能用部署共存作为未完成边界的补丁。

## 14. 验收矩阵

| 场景 | 必须成立 |
| --- | --- |
| 两 worker 同时唤醒 | 一次有效任务负责人 acquisition，一个活动 Runner |
| 同进程重复唤醒／启动失败 | 复用实例，或完整清理后重试，无任期来回替换 |
| PAUSED/PENDING 任务只收到 CANCEL | 不产生假 run，不经过假 RUNNING |
| M1 长执行，M2 到达 | 原任务负责人交付 M2，不创建第二个执行者 |
| MESSAGE 完成后长工具仍运行 | 后续合法控制可处理，命令队头不绑定工具耗时 |
| 前序文件准备／defer，后续 CANCEL | 顺序和延迟符合明确保留的协议，不虚假承诺立即取消 |
| checkpoint commit 后、回执前崩溃 | receipt 可恢复，不重复注入 |
| scheduled 提交后、启动前崩溃 | 可恢复启动，模型／工具尚未提前执行 |
| 首 checkpoint 后崩溃 | 使用实际恢复状态，不从原输入盲目重跑 |
| commit 回执丢失 | 读证据确认或保持未决，不立即二次执行 |
| checkpoint 修剪 | 已应用 receipt 不丢失 |
| 任务负责人接手后旧 callback 返回 | 执行与命令状态均不被旧任务负责人改写 |
| 同任务负责人内命令处理次数变化 | 旧 callback 不能完成新尝试 |
| 等待释放与入队交错／通知丢失 | 工作最终被扫描发现 |
| 旧 run 控制命令遇到新 run | 拒绝旧目标，不误控新执行 |
| 两个入口同时尝试恢复 | admission 只允许一个运行句柄 |
| 两个答案竞争 | 保留问题级 CAS 语义 |
| 停机后旧 processing 遗留 | 有证据补回执，未知效果不自动重放 |
| 连接池超时、关闭、反复取消 | 无忙循环、无悬空心跳、无旧任期释放新任期 |
| 授权撤销、外部 scope、文件绑定 | 继续遵守现有业务契约和可见性 |

数据库原子性与竞争用例覆盖 SQLite/PostgreSQL；跨 worker 场景至少包含真实多进程
和受控进程终止，不能全部用一个事件循环的 mocks 代替。再运行完整 Web CI，
尤其包括输出存储、历史失败重放、Workforce、checkpoint 和所有权夹具。

## 15. 范围估算与实施前检查

本设计选择了比早期估算更明确的任务租约与执行生命周期分离和持久效果凭据，
也包含其他入口的最小 admission 接入。预计 12–18 个生产文件、1–2 个新模块、
一个 migration、约 12–20 个测试文件；行数仍只能按约 1000–2000 行生产代码、
1500–3000 行测试估算，包含改写，不是净增加承诺。

停机部署省掉的是版本共存和任务归属分流，不能省略同一版本内部的入口隔离和恢复。

实施前必须完成以下函数级核对，若契约不成立，应修订设计而不是静默降级：

- 首个完整 checkpoint 能否在任何外部执行前提交，现有启动参数是否全部可恢复。
- checkpoint 写入与命令 receipt 是否可在同一事务、同一连接和一致锁序下完成。
- 所有调用 acquire/release/managed lease 的第一方入口是否都能接入共同任务负责人。
- 外部 executor 对处理次数和回执的依赖是否可以保留，未知效果是否正确阻止重放。

以上是实现前验证项，不是上线后再补的 TODO。只有全部成立、验收矩阵通过、
旧状态整理可重入且所有旧 worker 确认停止，B 才可以切换。
