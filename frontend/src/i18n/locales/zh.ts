const zh = {
  common: {
    back: "返回",
    loading: "加载中...",
    save: "保存",
    cancel: "取消",
    delete: "删除",
    confirm: "确定",
    deleteConfirm: "确定要删除这个任务吗？",
    notice: "提示",
    copy: "复制",
    copied: "已复制",
    send: "发送",
    pause: "暂停",
    resume: "继续",
    executing: "正在执行中",
    planning: "正在制定计划",
    taskPaused: "任务已暂停",
    noData: "暂无数据",
    errors: {
      unknown: "未知错误",
    },
  },
  nav: {
    settings: "系统设置",
    dashboard: "仪表盘",
    vibe: "Vibe",
    build: "Agents",
    models: "模型",
    files: "文件",
    memory: "记忆",
    monitoring: "监控",
    knowledgeBase: "知识库",
    knowledgeBaseDetail: "知识库详情",
    userManagement: "用户管理",
    login: "登录",
    register: "注册",
    agentStore: "Agent Store",
    tools: "工具",
    deploy: "部署",
    agentDev: "Agent 构建与部署",
    task: "任务",
    history: "全部任务",
    templates: "模板",
    sections: {
      agentDevelopment: "Agent 开发",
      resources: "资源",
    },
  },
  chatPage: {
    page: {
      emptyTitle: "描述目标，{appName} 会完成剩下的事",
      emptyDescription: "将您的想法自动转化为可运作的智能体。只需描述您的需求，即可快速启动",
      startWith: "你可以从这里开始",
    },
    cards: {
      createPPT: {
        title: "制作 PPT",
        description: "销售报告，幻灯片",
        prompt: "为 Q3 电商销售分析制作 PPT"
      },
      dataAnalysis: {
        title: "数据分析",
        description: "趋势，反馈",
        prompt: "分析客户反馈数据并总结趋势"
      },
      designPoster: {
        title: "设计海报",
        description: "社交媒体素材",
        prompt: "为即将到来的技术会议设计海报"
      },
      automatic: {
        title: "自动化",
        description: "自定义工作流",
        prompt: "构建一个自动邮件处理工作流"
      }
    },
    input: {
      placeholder: "描述您的任务...",
      hintEnter: "按 Enter 发送",
      processing: "处理中",
      actions: {
        config: "配置模型",
        upload: "上传文件",
      },
      noModel: "暂无模型",
      noModelAlert: "暂无模型，请先配置模型",
    },
    message: {
      assistantTitle: "AI 助手",
    },
    reasoning: {
      title: "思考过程",
      stepsCompleted: "已完成 {completed}/{total} 步",
      thinking: "正在思考",
      initializing: "正在初始化推理过程...",
      defaultStepTitle: "步骤 {index}",
    },
    executionPlan: {
      title: "执行计划",
      tooltip: "点击查看执行计划",
    },
    aiOutput: {
      title: "AI 输出",
      subtitle: "生成的响应",
      generating: "正在生成响应",
    },
    tokenUsage: {
      input: "输入tokens",
      output: "输出tokens",
    },
    fileUpload: {
      dropHere: "将文件拖拽到此处",
      uploadDocs: "上传文档",
      hintDragClick: "拖拽文件或点击选择",
    },
  },
  branding: {
    subtitle: "AI Agents for Everyone",
    tagline: `Run instantly.
Build when you need.`
  },
  settings: {
    title: "系统设置",
    description: "管理系统的基本配置和高级选项",
    language: {
      title: "语言设置",
      description: "选择界面显示语言",
    },
    versionControl: {
      title: "智能规划",
      description: "AI 自动拆解复杂任务并逐步执行"
    },
    team: {
      title: "长期记忆",
      description: "跨任务和对话保持关键上下文"
    },
    automation: {
      title: "人工监管",
      description: "随时审查、调整和引导 AI 执行"
    },
    password: {
      title: "修改密码",
      description: "更新您的账户密码",
      current: "当前密码",
      current_placeholder: "请输入当前密码",
      new: "新密码",
      new_placeholder: "请输入新密码",
      confirm: "确认新密码",
      confirm_placeholder: "请再次输入新密码",
      submit: "更新密码",
      submitting: "更新中...",
      errors: {
        fill_all: "请填写所有密码字段",
        mismatch: "新密码和确认密码不匹配",
        too_short: "新密码长度至少6位",
        network: "网络错误，请稍后重试",
      },
      success: "密码更新成功",
      failed: "密码更新失败",
    },
    defaultModels: {
      title: "默认模型设置",
      description: "为不同类型的任务设置默认使用的模型。这些设置仅对您个人有效。",
      types: {
        general: { title: "通用模型", description: "适用于大多数任务" },
        small_fast: { title: "小型快速模型", description: "轻量且响应迅速" },
        visual: { title: "视觉模型", description: "理解并处理图像" },
        compact: { title: "长上下文模型", description: "处理长上下文与压缩" },
        embedding: { title: "嵌入模型", description: "生成向量嵌入" },
      },
      actions: {
        clearDefault: "清除默认",
      },
      labels: {
        selectDefault: "选择默认模型",
        selectModel: "选择模型",
      },
      empty: {
        noModels: "暂无可用模型，请先在模型管理页面添加模型",
      },
      messages: {
        loadFailed: "加载模型配置失败",
        updated: "已更新：{type}",
        setFailed: "设置默认模型失败",
        removed: "已清除：{type}",
        removeFailed: "清除默认模型失败",
      },
      guide: {
        title: "使用说明",
        items: {
          personalOnly: "这些设置仅对您个人有效",
          fallbackFirstAvailable: "未设置默认时，系统将使用第一个可用模型",
          addOnModelsPage: "请先在模型管理页面添加模型，再进行选择",
          adminShared: "管理员可配置团队共享模型",
        },
      },
    },
  },
  register: {
    mobile_title: "创建账户",
    title: "注册 {appName}",
    description: "创建您的账户以开始使用智能体平台",
    form: {
      username: "用户名",
      username_placeholder: "请输入用户名",
      password: "密码",
      password_placeholder: "请输入密码（至少6位）",
      confirm_password: "确认密码",
      confirm_password_placeholder: "请再次输入密码",
      submit: "创建账户",
      submitting: "注册中...",
    },
    alerts: {
      success: "注册成功！请登录",
      failed: "注册失败",
      failed_retry: "注册失败，请稍后重试",
      password_mismatch: "密码不匹配",
      password_too_short: "密码长度至少6位",
    },
    login_hint: {
      has_account: "已有账户？",
      login_now: "立即登录",
    },
    status: {
      agent_running: "Agent 服务运行中",
      open_register: "开放注册",
    },
    features: {
      vbd: {
        title: "自主规划",
        description: "AI 先规划后执行。将复杂任务拆解并逐步执行。",
      },
      hitl: {
        title: "内置记忆",
        description: "记住重要信息。在任务和对话中保持上下文。",
      },
      timetravel: {
        title: "人工监管",
        description: "您来引导。AI 执行。随时审查、调整和介入。",
      },
    },
  },
  setup: {
    title: "初始化 {appName}",
    description: "请先创建第一个管理员账号以完成系统初始化",
    features: {
      bootstrap: {
        title: "初始化引导",
        description: "一步完成首次启动所需配置。",
      },
      config: {
        title: "快速进入配置",
        description: "初始化后可继续配置模型与工具能力。",
      },
      security: {
        title: "管理员优先",
        description: "先创建管理员账号，再开放系统访问。",
      },
    },
    form: {
      username: "管理员用户名",
      password: "管理员密码",
      confirmPassword: "确认密码",
      submit: "创建管理员",
      submitting: "创建中...",
    },
    errors: {
      required: "请填写用户名和密码",
      passwordTooShort: "密码至少 6 位",
      passwordMismatch: "两次输入的密码不一致",
      failed: "初始化失败",
    },
  },
  login: {
    mobile_title: "欢迎回来",
    title: "登录 {appName}",
    description: "请输入您的账户信息以访问智能体平台",
    form: {
      username: "用户名",
      username_placeholder: "请输入用户名",
      password: "密码",
      password_placeholder: "请输入密码",
      submit: "登录",
      submitting: "登录中...",
    },
    alerts: {
      auth_failed: "用户名或密码错误",
      network_failed: "登录失败，请稍后重试",
    },
    options: {
       remember_me: "记住登录状态",
       forgot_password: "忘记密码？",
     },
    register_prompt: "还没有账户？",
    register_link: "立即注册",
    register_closed: "当前已关闭注册",
     features: {
       version_control: {
         title: "自主规划",
         description: "AI 先规划后执行。将复杂任务拆解并逐步执行。",
       },
       team: {
         title: "内置记忆",
         description: "记住重要信息。在任务和对话中保持上下文。",
       },
       automation: {
         title: "人工监管",
         description: "您来引导。AI 执行。随时审查、调整和介入。",
       },
     },
   },
  vibe: {
    title: "Vibe 阶段",
    description: "把 prompts 变成可运行的 Agent，大模型自动规划，释放想象力",
    actions: {
      you_can: "你可以",
      new_task: "新建任务",
      delete_confirm: "确定要删除这个任务吗？",
    },
    quick_examples: {
      make_ppt: {
        title: "制作PPT",
        description: "季度业务回顾、营销提案",
      },
      data_analysis: {
        title: "数据分析",
        description: "电商销售、客户反馈",
      },
      design_poster: {
        title: "设计海报",
        description: "活动宣传、社交媒体",
      },
      automation: {
        title: "自动化",
        description: "邮件处理、文档管理",
      },
      more: {
        title: "更多",
        description: "探索无尽可能",
      },
    },
    list: {
      title: "任务列表",
      search_placeholder: "搜索任务...",
      empty: "暂无任务记录",
      pagination: {
        summary: "显示 {from}-{to} 条，共 {total} 条任务",
        prev: "上一页",
        next: "下一页",
      },
      status: {
        completed: "已完成",
        running: "运行中",
        failed: "失败",
        pending: "等待中",
        paused: "已暂停",
      },
      mode: {
        task: "任务",
        process: "流程",
      },
    },
  },
  userManagement: {
    title: "用户管理",
    description: "管理系统中的所有用户账户",
    stats: {
      total_users_prefix: "共",
      total_users_suffix: "个用户",
    },
    list: {
      title: "用户列表",
      description: "查看和管理系统中的所有用户",
      search_placeholder: "搜索用户名...",
      table: {
        id: "ID",
        username: "用户名",
        role: "角色",
        created_at: "创建时间",
        updated_at: "最后更新",
        actions: "操作",
        admin: "管理员",
        normal: "普通用户",
        delete: "删除",
        delete_confirm_title: "确认删除用户",
        delete_confirm_description_prefix: "您确定要删除用户 \"",
        delete_confirm_description_suffix: "\" 吗？此操作无法撤销，该用户的所有数据将被永久删除。",
        cancel: "取消",
        confirm_delete: "删除",
      },
      loading: {
        title: "加载中...",
        description: "正在验证用户权限",
      },
      no_permission: {
        title: "权限不足",
        description: "只有管理员可以访问用户管理页面",
      },
      alerts: {
        delete_success_prefix: "删除成功：用户 ",
        delete_success_suffix: " 已被删除",
        delete_failed_prefix: "删除失败：",
        delete_failed_suffix: "",
      },
      pagination: {
        summary: "显示 {from}-{to} 条，共 {total} 条用户",
        prev: "上一页",
        next: "下一页",
        page: "第 {current} 页，共 {total} 页",
      },
    },
  },
  dashboard: {
      stats: {
        total_tasks: "总任务数",
        active_agents: "活跃Agent",
        deployed_apps: "已部署应用",
        today_calls: "今日调用",
      },
      features: {
        title: "主要功能",
        vibe: {
          title: "VIBE",
          description: "创建智能Agent，自动规划，释放想象力",
        },
        build: {
          title: "BUILD",
          description: "将Agent工作流构建为可部署的应用",
        },
        deploy: {
          title: "DEPLOY",
          description: "部署应用到生产环境，实现业务价值",
        },
        models: {
          title: "模型管理",
          description: "配置和管理AI模型，设置默认模型",
        },
        monitoring: {
          title: "监控",
          description: "监控LLM调用、工具使用和系统性能",
        },
        files: {
          title: "文件管理",
          description: "上传、管理和查看项目文件",
        },
      },
      tasks: {
        title: "最近任务",
        search_placeholder: "搜索任务...",
        new_task: "新建任务",
        empty: "暂无任务记录",
        delete_confirm: "确定要删除这个任务吗？",
        status: {
          completed: "已完成",
          running: "运行中",
          failed: "失败",
          pending: "等待中",
        },
      },
      files: {
        title: "最近文件",
        view_all: "查看全部",
        empty: "暂无文件",
      },
  },
  monitoring: {
    title: "监控",
    description: "监控平台性能、LLM 调用和工具使用情况",
    cards: {
      totalCalls: "总调用次数",
      successRate: "成功率",
      avgResponseTime: "平均响应时间",
      todayCalls: "今日调用"
    },
    models: {
      title: "模型统计",
      empty: "暂无模型数据",
      summary: "成功率：{success}% • 总任务数：{total}",
      usageRate: "使用率"
    },
    tools: {
      title: "工具使用",
      empty: "暂无工具使用数据",
      avgDuration: "平均时长：{duration}s",
      usageCount: "使用次数 {count}"
    }
  },
  tools: {
    header: {
      title: "工具管理",
      description: "管理内置工具和 MCP 服务器配置",
    },
    tabs: {
      all: "全部工具",
      mcp: "MCP 服务",
    },
    stats: {
      mcpServers: {
        title: "MCP 服务器",
        activeCount: "{count} 个活跃服务器",
      },
      builtinTools: {
        title: "内置工具",
        summary: "{total} 个工具（{enabled} 个已启用）",
      },
      totalTools: {
        title: "总工具数",
        summary: "{total} 个工具（{enabled} 个已启用）",
      },
    },
    list: {
      title: "工具列表",
      description: "查看和管理所有可用的工具",
      searchPlaceholder: "搜索工具...",
      disabledBadge: "已禁用",
      sourcePrefix: "来源：{source}",
      actions: {
        enable: "启用",
        disable: "禁用",
      },
      empty: {
        title: "未找到工具",
        description: "尝试调整搜索条件或添加新工具",
      },
      noDescription: "未提供描述",
      transport: "传输协议",
      usedByAgents: "被 {count} 个任务使用",
    },
    mcp: {
      title: "MCP 服务器管理",
      description: "管理您的 Model Context Protocol (MCP) 服务器配置",
      addServer: "添加服务器",
      badge: "MCP 服务器",
      dialog: {
        addTitle: "添加 MCP 服务器",
        editTitle: "编辑 MCP 服务器",
        description: "配置 MCP 服务器连接参数",
      },
      form: {
        nameLabel: "服务器名称 *",
        namePlaceholder: "输入服务器名称",
        transportLabel: "传输协议 *",
        transportPlaceholder: "选择传输协议",
        descriptionLabel: "描述",
        descriptionPlaceholder: "输入服务器描述",
        fieldPlaceholderPrefix: "请输入 {field}",
      },
      transports: {
        stdio: { label: "标准输入输出", description: "通过标准输入输出流通信" },
        sse: { label: "服务器发送事件", description: "通过 Server-Sent Events 通信" },
        websocket: { label: "WebSocket", description: "通过 WebSocket 协议通信" },
        streamable_http: { label: "可流式 HTTP", description: "通过 HTTP 流式通信" },
      },
      buttons: {
        cancel: "取消",
        update: "更新",
        create: "创建",
        save: "保存",
      },
      empty: {
        title: "暂无 MCP 服务器配置",
        description: "点击\"添加服务器\"开始配置",
      },
      list: {
        defaultBadge: "默认",
        createdAt: "创建时间：{date}",
      },
      alerts: {
        nameRequired: "服务器名称不能为空",
        saveFailed: "保存失败",
        deleteFailed: "删除失败",
        toggleFailed: "切换状态失败",
      },
      delete: {
        confirm: "确定要删除服务器 \"{name}\" 吗？",
      },
    },
    categories: {
      development: "开发工具",
      search: "搜索工具",
      knowledge: "知识库",
      file: "文件操作",
      utility: "实用工具",
      creative: "创意工具",
      analysis: "分析工具",
      mcp: "MCP 工具",
    },
    badges: {
      types: {
        tool: "工具",
        basic: "基础",
        file: "文件",
        knowledge: "知识库",
        image: "图像",
        vision: "视觉",
        special_image: "图像",
      },
      source: {
        builtin: "内置",
      },
      basic: "基础",
      file: "文件",
      knowledge: "知识库",
      image: "图像",
      vision: "视觉",
      special_image: "图像",
      builtin: "内置",
    },
  },
  files: {
    header: {
      title: "文件管理",
      description: "管理项目文件：上传、预览、下载与删除",
    },
    search: {
      placeholder: "搜索文件...",
    },
    actions: {
      upload: "上传",
      uploading: "上传中...",
      delete: "删除",
      download: "下载",
      preview: "预览",
    },
    sidebar: {
      folders: "文件夹",
      allFiles: "所有文件",
      agents: "智能体",
      noAgents: "未找到智能体",
      system: "系统",
      userUploads: "用户上传",
    },
    breadcrumb: {
      files: "文件",
      unknownAgent: "未知智能体",
      unknownCategory: "未知分类",
    },
    selection: {
      selected: "已选择 {count} 项",
    },
    table: {
      name: "名称",
      type: "类型",
      size: "大小",
      dateModified: "修改日期",
      empty: {
        loading: "加载文件中...",
        noMatch: "没有匹配的文件。",
        noFiles: "未找到文件。",
      },
    },
    time: {
      justNow: "刚刚",
      minsAgo: "{count} 分钟前",
      hoursAgo: "{count} 小时前",
      daysAgo: "{count} 天前",
      monthsAgo: "{count} 个月前",
      yearsAgo: "{count} 年前",
    },
    delete: {
      confirmSingle: "确定要删除 {name} 吗？",
      confirmMultiple: "确定要删除 {count} 个文件吗？",
    },
    previewDialog: {
      buttons: {
        download: "下载",
        openInNewWindow: "新窗口打开",
      },
      loading: "加载文件内容...",
      imageError: {
        title: "图片加载失败",
        hint: "请尝试下载查看",
      },
      emptyContent: "文件内容为空",
      errors: {
        loadFailed: "文件加载失败",
        docxRenderFailed: "DOCX 预览渲染失败",
        cors: "CORS 错误：无法访问文件。这可能是浏览器缓存问题，请尝试刷新页面。",
        networkErrorWithMsg: "网络错误：{msg}",
      },
    },
  },
  memory: {
    loading: "正在加载记忆...",
    header: {
      backTitle: "返回首页",
      title: "记忆中心",
      description: "存储、搜索并管理智能体的记忆项",
      create: "新建记忆",
    },
    createDialog: {
      title: "创建记忆",
      description: "填写内容、关键词、标签和元数据以添加新的记忆项",
    },
    stats: {
      total: "记忆总数",
      categories: "类别数",
      tags: "标签数",
      storeType: "存储类型",
    },
    filters: {
      title: "筛选条件",
      category: {
        label: "类别",
        all: "全部",
        placeholder: "选择类别",
      },
      categoryOptions: {
        general: "通用",
        react_memory: "React 记忆",
        dag_plan_execute_memory: "DAG 规划执行记忆",
      },
      search: {
        label: "搜索",
        placeholder: "搜索内容、关键词或标签...",
      },
      similarity: {
        label: "相似度阈值",
        strict: "严格",
        lenient: "宽松",
      },
      date: {
        fromLabel: "开始时间",
        toLabel: "结束时间",
        fromPlaceholder: "开始日期",
        toPlaceholder: "结束日期",
      },
    },
    item: {
      keywordPrefix: "关键词",
      tagPrefix: "标签",
    },
    list: {
      empty: "暂无记忆项",
    },
    viewDialog: {
      title: "查看记忆",
      description: "查看选中记忆项的详细信息",
      labels: {
        category: "类别",
        time: "时间",
        content: "内容",
        keywords: "关键词",
        tags: "标签",
        metadata: "元数据",
      },
      close: "关闭",
      edit: "编辑",
    },
    editDialog: {
      title: "编辑记忆",
      description: "修改内容、关键词、标签或元数据",
    },
    deleteDialog: {
      title: "删除记忆",
      description: "确定要删除该记忆项吗？此操作不可撤销。",
      bulkDescription: "确定要删除 {count} 条记忆吗？此操作不可撤销。",
      delete: "删除",
    },
    form: {
      contentLabel: "内容",
      contentPlaceholder: "请输入记忆内容...",
      keywordsLabel: "关键词",
      keywordsPlaceholder: "请输入关键词，使用逗号分隔",
      tagsLabel: "标签",
      tagsPlaceholder: "请输入标签，使用逗号分隔",
      metadataLabel: "元数据（JSON）",
      metadataPlaceholder: "以 JSON 格式输入元数据，例如 {\"source\":\"web\"}",
    },
    sidebar: {
      contextSource: "上下文来源",
      allMemories: "所有记忆",
      byCategory: "按类别",
    },
    actions: {
      exportSelected: "导出选中 ({count})",
      exportCsv: "导出 CSV",
      deleteSelected: "删除选中 ({count})",
      clearFilter: "清除筛选",
    },
    empty: {
      title: "未找到记忆",
      searchHint: "尝试调整搜索关键词。",
      categoryHint: "此类别下暂无记忆。",
    },
    errors: {
      deleteFailed: "删除记忆失败",
      deleteBatchFailed: "批量删除记忆失败",
      createFailed: "创建记忆失败",
      updateFailed: "更新记忆失败",
      unknown: "未知错误",
    },
  },
  templates: {
    title: "模板中心",
    subtitle: "使用预构建的智能体快速启动工作流。",
    searchPlaceholder: "搜索模板...",
    useTemplate: "使用模板",
    noResults: "未找到匹配的模板。",
    categoryTitles: {
      all: "全部",
      featured: "推荐模板",
      healthcare_fitness: "医疗与健康",
      general_productivity: "通用与效率",
      customer_service: "客户服务",
      finance_lms_ops: "金融、LMS 与运营",
      security: "安全"
    },
    sections: {
      knowledge: "知识与研究",
      sales: "销售与外联"
    },
    items: {
      internal_qa: {
        title: "内部问答",
        description: "连接内部文档并自动回答员工问题。"
      },
      deep_research: {
        title: "深度研究",
        description: "研究公司、网页和报告以汇编综合数据。"
      },
      outbound_bdr: {
        title: "外联 BDR",
        description: "抓取潜在客户并起草个性化的外联邮件。"
      },
      inbound_lead: {
        title: "入站线索处理",
        description: "资格审查入站线索并自动安排会议。"
      },
      crm_enrichment: {
        title: "CRM 丰富",
        description: "利用网络和 LinkedIn 数据丰富 CRM 联系人信息。"
      },
      sales_call_notes: {
        title: "销售通话记录",
        description: "转录并总结销售通话以便跟进。"
      },
      invoice_automation: {
        title: "发票自动化",
        description: "从发票中提取数据并更新会计系统。"
      }
    }
  },
  sidebar: {
    tasks: {
      empty: "暂无任务",
      viewAll: "查看全部",
    },
    user: {
      loggedIn: "已登录",
      notLoggedIn: "未登录",
      logoutTitle: "退出登录",
      defaultName: "用户",
    },
  },
  agentStore: {
    title: "Agent Store",
    subtitle: "发现和使用专业的 AI Agent，覆盖数据分析、代码生成、文本处理等多个领域",
    builtIn: {
      title: "内置 Agents",
      badgeOfficial: "官方",
    },
    text2sql: {
      title: "问数助手",
      descriptionShort: "智能数据库查询助手，将自然语言转换为 SQL",
      featuresParagraph: "支持多种数据库类型，提供智能模式分析、SQL 生成优化、结果格式化等功能。适用于数据分析、报表生成、业务查询等场景。",
      startUsing: "开始使用",
      config: {
        subtitle: "选择数据库连接开始查询",
        connectionsTitle: "数据库连接",
        startQuery: "开始查询",
        noConnections: "暂无数据库连接，请在下方添加新数据库",
        readonlyBadge: "只读",
        addNewDatabaseTitle: "添加新数据库",
        addDatabase: "添加数据库",
        adding: "正在添加...",
        form: {
          nameLabel: "数据库名称",
          namePlaceholder: "请输入数据库名称，例如：生产库",
          typeLabel: "数据库类型",
          urlLabel: "连接字符串",
          urlPlaceholder: "请输入数据库连接字符串",
          readonlyLabel: "以只读模式连接（推荐）",
        },
        alerts: {
          fillAll: "请填写所有必填信息",
          addFailedPrefix: "添加数据库失败：",
          unknownError: "未知错误",
          addFailed: "添加数据库失败，请稍后重试",
          testSuccess: "连接测试成功：{message}",
          testFailedPrefix: "连接测试失败：",
          testFailed: "连接测试失败，请稍后重试",
        },
        statusBadge: {
          connected: "已连接",
          disconnected: "未连接",
          error: "错误",
        },
        loginRequiredTitle: "请先登录",
        loginRequiredDescription: "需要登录才能使用问数助手",
      },
      chat: {
        login: {
          title: "请先登录",
          description: "需要登录才能使用 Text2SQL Agent",
        },
        database: {
          unknown: "未知数据库",
        },
        connection: {
          connected: "已连接",
          connecting: "连接中",
          initializing: "正在初始化连接...",
          wsConnecting: "正在建立 WebSocket 连接...",
          ready: "连接已就绪",
        },
        history: {
          title: "历史任务",
          newConversation: "开始新对话",
          emptyTitle: "暂无历史任务",
          emptyDescription: "创建新任务开始查询数据库",
          delete: "删除任务",
          deleteConfirm: "确定要删除这个任务吗？",
          deleteFailed: "删除任务失败",
        },
        mode: {
          readOnly: "只读模式",
          readWrite: "读写模式",
        },
        initializing: "正在初始化问数助手...",
        welcome: {
          title: "欢迎使用问数助手",
          description: "请选择左侧的历史对话，或在下方直接输入消息开始查询",
          featuresTitle: "问数助手可以帮助您：",
          features: {
            naturalToSql: "将自然语言转换为 SQL 查询",
            analyzeSchema: "分析数据库表结构",
            executeAndShow: "执行 SQL 查询并展示结果",
            optimizeQuery: "提供查询优化建议",
          },
        },
        input: {
          title: "请输入您的查询需求",
          hint: "提示：助手会自动分析数据库结构并生成相应的 SQL 查询",
          placeholder: "请输入您的查询需求...",
        },
        status: {
          ready: "准备就绪",
          connecting: "连接中...",
          initializing: "初始化中...",
        },
        aiSummary: {
          title: "AI 总结",
        },
        table: {
          title: "查询结果",
          titleWithCount: "查询结果 ({count} 条记录)",
          buttons: {
            chart: "图表",
            bar: "柱状图",
            pie: "饼图",
            line: "折线图",
            copyData: "复制数据",
            exportCsv: "导出 CSV",
            copy: "复制",
          },
        },
        steps: {
          analyzeSchema: "数据库结构分析",
          executeSql: "SQL查询执行",
          formatResult: "结果格式化",
          defaultStep: "查询处理步骤",
          plan: "查询策略规划",
        },
        thinking: {
          schemaCompleted: "数据库表结构分析完成",
          schemaRunning: "正在分析数据库表结构和关系...",
          sqlGenCompleted: "SQL查询语句生成完成",
          sqlGenRunning: "正在生成和优化SQL查询语句...",
          executeCompleted: "SQL查询执行完成",
          executeRunning: "正在执行SQL查询...",
          formatCompleted: "查询结果格式化完成",
          formatRunning: "正在整理和格式化查询结果...",
          planCompleted: "查询策略制定完成",
          planRunning: "正在制定查询策略和执行计划...",
          analysisCompleted: "问题分析完成",
          analysisRunning: "正在分析问题和数据库结构...",
          sqlStatementCompleted: "SQL语句生成完成",
          sqlStatementRunning: "正在生成SQL查询语句...",
          queryCompleted: "查询执行完成",
          queryRunning: "正在执行查询操作...",
          stepCompleted: "查询步骤处理完成",
          stepRunning: "正在处理查询步骤...",
        },
      },
      timeline: {
        title: "思考过程",
        stepType: {
          planning: "思考分析",
          analysis: "数据探索",
          sql_generation: "生成SQL",
          execution: "执行查询",
          result: "结果整理"
        },
        status: {
          pending: "等待中",
          running: "执行中",
          completed: "已完成",
          failed: "失败"
        },
        details: {
          sql: "生成的SQL",
          result: "查询结果",
          error: "错误信息"
        },
        actions: {
          cancelAutoCollapse: "取消自动折叠",
          toggleStep: "展开/收起步骤：{name}"
        },
        labels: {
          dependencies: "依赖：{count}",
          unknownStatus: "未知状态",
          basedOnSteps: "基于步骤：{steps}"
        }
      },
      charts: {
        noNumericData: "没有适合图表的数值数据",
        switchToTable: "切换到表格",
        bar: {
          rankTop: "{valueColumn} 排行 (前{top}条)",
        },
        pie: {
          distributionTop: "{valueColumn} 分布图 (前{top}条)",
        },
        line: {
          trendTop: "{valueColumn} 趋势图 (前{top}条)",
        },
        predict: {
          buttonPredict: "AI预测",
          buttonPredicting: "预测中...",
          successDefault: "预测数据生成成功",
          failedDefault: "预测生成失败",
          requestFailedHttp: "预测请求失败：HTTP {status}",
          requestFailedNetwork: "预测请求失败：无法连接到预测服务",
        },
        analysis: {
          trendAnalysisLabel: "趋势分析：",
          confidence: "置信度：{confidence}",
        },
        legend: {
          historical: "历史数据",
          prediction: "预测数据",
        },
      },
    },
    stats: {
      users: "{count} 用户",
      usesToday: "今日 {count} 次使用",
    },
    badges: {
      smartCorrection: "智能纠错",
    },
    moreBuiltIn: {
      title: "更多内置 Agents",
      comingSoon: "敬请期待...",
      description: "我们正在开发更多专业的 AI Agent，包括代码助手、数据分析、文档处理等。",
    },
    deployments: {
      title: "VIBD 部署的 Agents",
      badgeCommunity: "社区",
      noneTitle: "暂无部署的 Agents",
      noneDescription: "当您通过 VIBE → BUILD → DEPLOY 流程部署自定义 Agents 后，它们将显示在这里。",
      helperText: "使用我们的 VIBE 系统构建和部署您自己的 AI Agent",
    },
    actions: {
      createAgent: "开始创建 Agent",
    },
    footer: {
      title: "开始使用 AI Agent",
      description: "选择适合您需求的 Agent，立即开始智能工作流程",
      tryText2SQL: "体验 Text2SQL",
      createCustomAgent: "创建自定义 Agent",
    },
  },
  models: {
    header: {
      title: "模型管理",
      description: "配置和管理 AI 模型，设置默认模型",
      add: "添加模型",
    },
    search: {
      placeholder: "搜索模型...",
    },
    tabs: {
      llm: "LLM 模型",
      embedding: "Embedding 模型",
      image: "图像生成模型",
    },
    section: {
      enabledModels: "已启用模型",
      configuredCount: "已配置 {count} 个",
      exploreProviders: "模型供应商",
    },
    deleteConfirm: "确定要删除这个模型吗？",
    deleteTitle: "删除模型",
    errors: {
      setDefaultFailed: "设置默认模型失败",
      removeDefaultFailed: "移除默认模型失败",
    },
    defaults: {
      general: "默认模型",
      fast: "快速模型",
      visual: "视觉模型",
      compact: "长上下文模型",
      embedding: "嵌入模型",
      image: "图像生成模型",
      image_edit: "图像编辑模型",
      shared: "已分享",
    },
    card: {
      owner: "我的模型",
      shared: "共享",
      inactive: "未激活",
      fields: {
        provider: "提供商",
        name: "名称",
        temperature: "温度",
        not_set: "未设置",
        dimension: "向量维度",
        base_url: "基础 URL",
        default: "默认",
        abilities: "能力",
        description: "描述",
        created: "创建时间",
        updated: "更新时间",
        unknown: "未知",
      },
      actions: {
        edit: "编辑",
        setDefault: "设为默认",
        connect: "连接",
      },
      status: {
        connected: "已连接",
      },
    },
    empty: {
      llm: "暂无配置的 LLM 模型",
      embedding: "暂无配置的嵌入模型",
      image: "暂无配置的图像模型",
      description: "点击 '添加模型' 开始配置",
    },
    dialog: {
      connectProviderTitle: "连接 {provider}",
      connectProviderDescription: "输入您的 API Key 以获取可用模型。",
      fetchingModels: "正在获取模型...",
      modelsFetchedAfterConnect: "连接后将自动获取可用模型。",
      cancel: "取消",
      add: "创建模型",
      editTitle: "编辑模型",
      createTitle: "创建新模型",
      editDescription: "修改现有模型的配置信息",
      createDescription: "配置新模型以在系统中使用",
      providerModelsTitle: "{provider} 模型",
      manageDescription: "管理此提供商的已配置模型。",
      noModelsConfigured: "暂无配置的模型。",
      availableModels: "可用模型",
    },
    form: {
      modelId: "模型 ID",
      category: "模型类别",
      provider: "模型类型",
      name: "模型名称",
      temperature: "温度",
      dimension: "向量维度",
      apiKey: "API 密钥",
      apiKeyPlaceholder: "sk-...",
      apiKeyPlaceholderEdit: "留空以保持不变",
      baseUrl: "基础 URL",
      baseUrlPlaceholder: "https://api.openai.com/v1",
      abilities: "能力选择",
      description: "描述",
      setDefault: "设为默认模型",
      defaultPlaceholder: "选择默认类型...",
      shareWithUsers: "与所有用户共享此模型",
      abilitiesPlaceholder: "选择能力...",
      update: "更新模型",
      create: "创建模型",
    },
    abilities: {
      chat: "对话",
      vision: "视觉",
      tool_calling: "工具调用",
      thinking_mode: "思考模式",
      embedding: "嵌入",
      generate: "生成",
      edit: "编辑",
    },
    defaultDialog: {
      title: "设置默认模型",
      description: "为 {modelId} 设置默认模型类型",
      selectType: "选择模型类型",
      current: "当前",
      removeTitle: "移除默认",
      cancel: {
        general: "取消默认",
        fast: "取消快速",
        visual: "取消视觉",
        compact: "取消长上下文",
        embedding: "取消嵌入",
      },
      close: "关闭",
    },
    providers: {
      openai: {
        description: "访问 GPT-4o、GPT-4 Turbo 和 DALL-E 3 模型。推理和创造力的行业标准。",
      },
      azure_openai: {
        description: "Azure 托管的企业级 OpenAI 模型。",
      },
      zhipu: {
        description: "针对中英文任务优化的 GLM-4 模型。具有强大的推理能力。",
      },
      dashscope: {
        description: "来自阿里云的通义千问模型。在中文语言任务中表现出色。",
      },
      gemini: {
        description: "原生内置多模态能力。非常适合分析图像、视频和大型文本语料库。",
      },
      claude: {
        description: "Claude 3.5 Sonnet 和 Opus 模型。以高可控性、大上下文窗口和编码任务而闻名。",
      },
      xinference: {
        description: "本地推理平台，支持运行开源 LLM、嵌入模型和图像模型。在自己的基础设施上部署模型。",
      },
      openai_embedding: {
        description: "OpenAI 的专用嵌入模型。",
      },
    },
  },
  kb: {
    errors: {
      loadFailed: "知识库加载失败",
      nameRequired: "请输入知识库名称",
      createFailed: "创建知识库失败",
      uploadFileRequired: "请选择要上传的文件",
      uploadFailedFile: "上传文件失败: {name}",
      uploadFailed: "上传失败",
      startUrlRequired: "请输入起始 URL",
      webIngestFailed: "网站导入失败",
      deleteFailed: "删除知识库失败: {name}",
      deleteFailedGeneric: "删除知识库失败",
      unsupportedFileType: "不支持的文件类型",
    },
    actions: {
      deleteConfirm: "确定要删除知识库 {name} 吗？此操作无法撤销。",
    },
    loading: {
      loadingKB: "正在加载知识库...",
    },
    header: {
      title: "知识库管理",
      description: "管理文档集合和搜索索引",
      new: "新建知识库",
    },
    search: {
      placeholder: "搜索知识库...",
    },
    card: {
      documentsLabel: "文档",
      chunksLabel: "文本块",
      embeddingsLabel: "向量",
      parsesLabel: "解析",
      noDescription: "暂无描述",
      actions: {
        viewDetail: "查看详情",
      },
      status: {
        active: "活跃",
      },
    },
    empty: {
      noKB: "暂无知识库",
      searchNoMatch: "没有找到匹配的知识库",
      hintSearch: "尝试使用其他搜索关键词",
      hintCreate: "点击 '新建知识库' 开始创建您的第一个知识库",
    },
    dialog: {
      createTitle: "新建知识库",
      createDescription: "创建新的知识库并上传文档或从网站导入内容",
      basicInfo: {
        title: "基本信息",
        nameLabel: "知识库名称",
        namePlaceholder: "留空将使用文件名或 URL 作为知识库名称",
        descriptionLabel: "描述",
        descriptionPlaceholder: "知识库描述（可选）",
      },
      tabs: {
        file: "文件上传",
        web: "网站导入",
      },
      fileUpload: {
        title: "文档上传",
        dropOrClick: "拖拽文件到此处或点击选择",
        supportedFormats: "支持的格式：PDF、TXT、HTML、MD、Word、Excel、PPT等",
        selectFiles: "选择文件",
        selectedTitle: "已选择的文件",
        remove: "移除",
        uploading: "正在上传...",
        uploadSuccess: "上传成功",
        uploadFailed: "上传失败",
        progressTitle: "上传进度",
        processing: "处理中...",
        processSuccess: "处理成功",
        processResult: {
          createDocuments: "创建文档",
          textChunks: "文本块",
          vectors: "向量",
        },
      },
      webImport: {
        title: "网站导入",
        description: "遵守 robots.txt，爬取目标网站内容并建立索引",
        basic: {
          title: "基础配置",
          startUrl: "起始 URL",
          maxPages: "最大页面数",
          crawlDepth: "爬取深度",
          concurrentRequests: "并发请求数",
        },
        advanced: {
          title: "高级配置",
          urlPatterns: "URL 匹配规则",
          excludePatterns: "排除规则",
          contentSelector: "内容选择器",
          removeSelectors: "移除选择器",
          requestDelay: "请求间隔（毫秒）",
          requestDelaySeconds: "请求延迟（秒）",
          timeout: "请求超时（毫秒）",
          timeoutSeconds: "超时时间（秒）",
          sameDomainOnly: "仅同域名",
          respectRobotsTxt: "遵守 robots.txt",
          hintMultiple: "多个模式用逗号分隔",
          hintContentSelector: "CSS 选择器，用于提取主内容",
        },
        status: {
          progressTitle: "爬取进度",
          crawling: "正在爬取网站，这可能需要几分钟...",
          success: "爬取成功",
          done: "爬取完成",
        },
        result: {
          pages: "爬取页面",
          viewWarnings: "查看警告",
        },
      },
    },
    index: {
      title: "索引配置",
      parseMethod: "解析方法",
      parseOptions: {
        default: "默认",
        pypdf: "PyPDF",
        pdfplumber: "PDFPlumber",
        unstructured: "Unstructured",
        pymupdf: "PyMuPDF",
        deepdoc: "DeepDoc",
      },
      chunkStrategy: "分块策略",
      chunkOptions: {
        recursive: "递归",
        fixed_size: "固定大小",
        markdown: "Markdown",
      },
      chunkSize: "分块大小",
      chunkOverlap: "分块重叠",
      embeddingModelId: "嵌入模型",
      embeddingBatchSize: "批处理大小",
      startImport: "开始导入",
      saveConfig: "保存配置",
    },
    detail: {
      viewingDetails: "正在查看知识库 {name} 的详情",
      loadingDetail: "正在加载知识库详情...",
      notFound: "知识库未找到",
      backToList: "返回知识库列表",
      tabs: {
        files: "文件管理",
        search: "搜索测试",
        settings: "索引设置",
      },
      files: {
        title: "文件列表",
        addSource: "添加来源",
        addDialogTitle: "添加数据源",
        addDialogDescription: "选择要添加到此知识库的数据源",
        uploading: "正在上传...",
        upload: "开始上传",
      },
      uploaded: {
        title: "已上传文件",
        indexed: "已索引",
        emptyTitle: "暂无文件",
        emptyHint: "上传文件后将在此处显示",
        delete: "删除文档",
        confirmDelete: "确定要删除这个文档吗？",
      },
      process: {
        title: "处理结果",
        labels: {
          file: "文件",
          document: "文档",
          chunk: "块",
          parse: "解析",
          vector: "向量",
          error: "错误",
        },
      },
      search: {
        title: "搜索测试",
        resultsTitle: "搜索结果 ({count} 条)",
        score: "评分",
        metadata: "元数据",
        noResults: "未找到相关结果",
        typeLabel: "搜索类型",
        topKLabel: "返回数量",
        embeddingModelIdLabel: "嵌入模型",
        rerankModelIdLabel: "重排序模型",
        rerankPlaceholder: "可选",
        queryPlaceholder: "输入搜索内容...",
        searchButton: "搜索",
        types: {
          hybrid: "混合搜索",
          dense: "向量搜索",
          sparse: "稀疏搜索"
        }
      },
      edit: {
        title: "编辑知识库",
        description: "修改知识库名称",
        nameLabel: "知识库名称",
        namePlaceholder: "输入新的知识库名称",
        updating: "更新中...",
        errors: {
          renameFailed: "重命名知识库失败",
          updateFailed: "更新知识库失败",
        },
      },
      errors: {
        deleteFailed: "删除文档失败",
        pleaseSelectFiles: "请先选择文件",
        uploadFailedWithName: "上传文件失败: {name}",
        uploadFailedGeneric: "上传失败",
        enterStartUrl: "请输入起始 URL",
        webImportFailed: "网站导入失败",
        searchFailed: "搜索失败",
        partialSuccess: "部分成功: {message}",
        deleteFailedWithMessage: "删除失败: {message}",
      },
    },
  },
  build: {
    header: {
      backTitle: "返回到 Builds 列表",
      title: "Build",
      subTitle: "构建你的应用",
      status: {
        building: "构建中",
        completed: "已完成",
        failed: "失败"
      },
      deploy: {
        button: "部署",
        alertCompleteFirst: "请先完成一次构建再进行部署"
      }
    },
    page: {
      headerTitle: "Build 阶段",
      headerDescription: "将Vibe阶段的交互式工作流转换为可部署的应用",
      statusTitle: "构建状态",
      statusBadgeDev: "开发中",
      statusDevNote: "Build阶段正在开发中，未来将支持从Vibe阶段的交互式工作流自动生成可部署的应用代码。",
      settingsButton: "构建设置",
      stepsTitle: "构建步骤",
      steps: {
        codeGeneration: {
          title: "代码生成",
          description: "基于Vibe阶段生成可执行代码"
        },
        configBuild: {
          title: "配置构建",
          description: "设置应用配置和依赖项"
        },
        packageDeploy: {
          title: "打包部署",
          description: "打包应用并准备部署"
        },
        apiGeneration: {
          title: "API生成",
          description: "生成RESTful API接口"
        },
        status: {
          pending: "待开发"
        }
      },
      featuresTitle: "计划功能",
      features: {
        autoCode: {
          title: "自动代码生成",
          description: "从DAG工作流生成可执行代码"
        },
        multiPlatform: {
          title: "多平台支持",
          description: "支持Web、API、小程序等多种部署目标"
        },
        smartConfig: {
          title: "智能配置",
          description: "自动生成配置文件和依赖项"
        }
      }
    },
    left: {
      title: "配置输入",
      appTypeLabel: "应用类型",
      labels: {
        relatedTask: "关联的 Task",
        selectTask: "选择构建的 Task",
        loading: "加载中...",
        noTasksPrefix: "还没有 Task，请先",
        createTask: "创建 Task",
        goComplete: "前往完成"
      },
      hints: {
        cannotBuildIncomplete: "无法构建未完成的 Task。请先在 VIBE 中完成该任务。"
      }
    },
    actions: {
      cancel: "取消构建",
      viewHistory: "查看历史",
      start: "开始构建"
    },
    inputs: {
      title: "应用输入配置",
      addText: "文本",
      addFile: "文件",
      emptyHint: "点击上方按钮添加输入字段",
      descriptionPlaceholder: "描述这个参数",
      required: "必填"
    },
    model: {
      title: "构建模型",
      loading: "加载模型列表...",
      empty: "没有可用的 LLM 模型，请先在模型管理中添加模型",
      defaultSuffix: " (默认)"
    },
    process: {
      title: "构建过程",
      tdd: {
        title: "TDD 迭代进度",
        attempt: "尝试 {attempt}/3"
      },
      logs: {
        title: "构建日志",
        emptyHint: "开始构建后将显示日志",
        building: "构建中..."
      }
    },
    yaml: {
      title: "生成的 YAML"
    },
    messages: {
      loadingHistory: "加载历史构建记录",
      loadHistoryFailedPrefix: "加载历史记录失败: {error}",
      startingBuild: "开始构建应用...",
      buildStarted: "构建任务已启动 (Build #{number})",
      agentStarted: "Agent 开始执行...",
      execStepPrefix: "执行步骤: ",
      analyzing: "分析中",
      invokeToolPrefix: "调用工具: ",
      toolResultReceived: "工具结果已接收",
      completed: "构建完成！YAML workflow 已生成",
      failedPrefix: "构建失败: {error}",
      cancelled: "构建已被取消",
      cancelling: "正在取消构建...",
      inputs: {
        configureFirst: "请先在左侧配置输入字段",
      },
      tdd: {
        success: "TDD 尝试 {attempts} 成功！YAML 已生成",
        failed: "TDD 尝试 {attempts} 失败: {error}",
      },
      llm: {
        calling: "LLM 调用中...",
        responseReceived: "LLM 响应已接收",
      },
      ws: {
        noToken: "No authentication token found for WebSocket",
        connected: "已连接到构建服务器",
        error: "WebSocket 连接错误",
        closed: "WebSocket 连接已关闭",
      },
    },
    test: {
      title: "测试区域",
      ready: "就绪",
      notBuilt: "未构建",
      hintReady: "测试生成的 YAML workflow",
      hintNotReady: "请先构建应用以启用测试功能",
      waiting: "等待构建完成",
      waitingHint: "构建完成后即可在此测试应用功能",
      testing: "正在测试...",
      testingHint: "使用生成的 YAML 执行应用",
      result: {
        success: "测试成功",
        failed: "测试失败"
      },
      executionLog: "执行日志",
      finalResult: "最终结果",
      errorInfo: "错误信息",
      inputHint: "在下方输入参数并点击测试",
      input: {
        title: "测试输入",
        placeholderPrefix: "输入",
        submit: "测试"
      },
      messages: {
        successLog: "测试成功！执行了 {nodes} 个节点，耗时 {seconds} 秒",
        failedLog: "测试失败: {error}",
        execError: "测试执行失败: {message}"
      },
      deployTitle: "部署此应用",
      deployHint: "部署按钮需要：success={success}，buildId={id}"
    },
    appTypes: {
      website: "网站",
      miniprogram: "微信小程序",
      api: "API"
    }
  },
  builds: {
    header: {
      title: "构建历史",
      description: "查看和管理构建历史记录",
      newBuild: "新建 Build"
    },
    filters: {
      search_placeholder: "搜索 Build ID 或 Task 名称...",
      status: {
        all: "所有状态"
      }
    },
    stats: {
      total: "共 {count} 个构建"
    },
    status: {
      started: "已启动",
      building: "构建中",
      completed: "已完成",
      failed: "失败",
      cancelled: "已取消"
    },
    empty: {
      none: "还没有任何构建记录",
      createFirst: "创建第一个 Build",
      noMatch: "没有找到匹配的构建记录"
    },
    card: {
      appType: "应用类型",
      createdAt: "创建时间",
      completedAt: "完成时间",
      taskFallback: "Task #{id}",
      taskIdLabel: "Task ID",
      viewDetail: "查看详情",
      delete: "删除",
      deleteConfirm: "确定要删除这个 Build 吗？"
    },
    list: {
      header: {
        title: "我的 Agent",
        description: "管理你的自定义 Agent",
        create: "创建 Agent",
      },
      search: {
        placeholder: "搜索 Agent...",
      },
      empty: {
        title: "暂无 Agent",
        description: "开始创建你的第一个 Agent",
        create: "创建 Agent",
      },
      pagination: {
        summary: "显示第 {from} 到 {to} 条，共 {total} 条",
        prev: "上一页",
        next: "下一页",
      },
      status: {
        draft: "草稿",
        published: "已发布",
      },
      actions: {
        chat: "聊天",
        unpublish: "取消发布",
        edit: "编辑",
        deleteConfirm: "确定要删除这个 Agent 吗？",
      },
      chat: {
        welcome: "你好！我是 {name}。{description}\n\n你可以直接开始对话，或者选择下方的建议提示词。",
        inputPlaceholder: "输入你的消息...",
        loading: "加载 agent 中...",
        notFound: "Agent 未找到",
        failed: "加载 agent 失败",
        taskCreated: "任务已创建。正在实现 WebSocket 连接...",
        error: "错误：{message}",
        sendFailed: "发送消息失败，请重试。",
      }
    },
    pagination: {
      summary: "显示第 {from} 到 {to} 条，共 {total} 条",
      prev: "上一页",
      next: "下一页"
    },
    editor: {
      header: {
        title: "自定义 Agent",
        subtitle: "配置你的 AI Agent 的行为和能力。",
        create: "创建",
        update: "更新",
        publish: "发布",
        unpublish: "取消发布",
        creating: "创建中...",
        updating: "更新中...",
      },
      validation: {
        nameRequired: "Agent 名称不能为空",
        modelRequired: "请选择通用模型",
      },
      success: {
        created: "创建成功",
        createdDesc: "你的 '{name}' 已成功创建",
      },
      error: {
        failed: "创建 Agent 失败",
        unknown: "发生未知错误",
      }
    },
    configForm: {
      logo: {
        label: "Logo",
        upload: "上传 Logo",
        remove: "移除 Logo",
      },
      name: {
        label: "名称",
        placeholder: "例如：代码助手",
      },
      description: {
        label: "描述",
        placeholder: "描述该 Agent 的功能...",
      },
      instructions: {
        label: "指令",
        placeholder: "Agent 的系统指令...",
      },
      executionMode: {
        label: "执行模式",
        react: {
          title: "标准模式（推荐）",
          description: "快速响应，灵活处理。适合大多数日常任务。",
        },
        graph: {
          title: "规划模式",
          description: "系统化拆解，步步为营。适合复杂的多步骤任务。",
        },
      },
      model: {
        label: "模型",
        noData: "暂无可用模型",
        types: {
          general: "通用模型",
          smallFast: "快速模型",
          visual: "视觉模型",
          compact: "紧凑模型",
        },
        tips: {
          general: "适用于大多数任务的通用大语言模型，平衡了性能和成本。",
          smallFast: "响应速度快，成本较低，适用于简单任务或需要低延迟的场景。",
          visual: "具备图像识别和处理能力，适用于涉及图片的任务。",
          compact: "轻量级模型，适用于资源受限或对即时性要求极高的场景。",
        },
      },
      knowledgeBase: {
        label: "知识库",
        placeholder: "选择知识库",
        create: "创建新知识库",
        noData: "暂无可用知识库",
        selected: "已选择 {count} 个",
      },
      skills: {
        label: "技能",
        placeholder: "选择技能",
        create: "创建新技能",
        noData: "暂无可用技能",
      },
      tools: {
        label: "工具",
        placeholder: "选择工具类别",
        noData: "暂无可用工具",
        selectedCount: "已选择 {count} 个类别（共 {tools} 个工具）",
        categories: {
          basic: "基础工具",
          file: "文件操作",
          vision: "视觉",
          image: "图像生成",
          knowledge: "知识库",
          mcp: "MCP 工具",
          browser: "浏览器自动化",
          ppt: "PowerPoint",
          office: "办公工具",
          specialImage: "图像工具",
        },
      },
      suggestedPrompts: {
        label: "建议提示词",
        description: "为用户提供一些示例提示词，让他们可以直接点击使用",
        placeholder: "示例提示词 {index}",
        add: "+ 添加提示词",
        delete: "删除",
      },
      actions: {
        save: "保存配置",
        reset: "重置",
      }
    },
    preview: {
      title: "预览",
      initialMessage: "你好！我是你的 Agent 预览。你可以在这里测试你的配置。",
      chat: {
        placeholder: "与你的 Agent 对话...",
        send: "发送",
        you: "你",
        agent: "测试 Agent",
      },
      errors: {
        noModel: "请先选择通用模型",
        requestFailed: "预览请求失败，请检查网络连接",
      }
    }
  },
  agent: {
    header: {
      backTitle: "返回Vibe",
      badge: {
        process: "流程",
        task: "任务",
      },
      connection: {
        connected: "已连接",
        disconnected: "未连接",
        errorPrefix: "连接错误: {message}",
      },
      buildButton: "构建应用",
    },
    build: {
      onlyProcessMode: "仅流程模式可构建应用",
      onlyProcessModeShort: "仅流程模式可构建",
    },
    initial: {
      title: "开始你的任务",
      subtitle: "选择模式并描述你想要完成的任务，AI 将为你制定计划并执行",
    },
    input: {
      placeholder: {
        default: "详细描述你的任务需求...",
        continueExecution: "输入调整指令，修改执行计划...",
      },
      actions: {
        config: "配置",
        uploadFile: "上传文件",
        sendWithFiles: "发送并上传文件",
        sendMessage: "发送消息",
        pauseTask: "暂停任务",
        resumeTask: "继续任务",
      },
      status: {
        labels: {
          running: "运行中",
          completed: "已完成",
          failed: "失败",
          pending: "等待中",
          paused: "已暂停",
        },
        selectedFiles: "已选择 {count} 个文件",
        taskStatusPrefix: "任务状态: ",
        processing: "处理中...",
      },
    },
    config: {
      title: "配置模型",
    },
    configDialog: {
      title: "Agent 配置",
      description: "配置 Agent 的模型参数，更改将立即生效（仅当前会话）",
      modelSelect: {
        label: "选择模型",
        description: "从已添加的模型中选择用于处理任务的 AI 模型",
        loading: "加载模型列表...",
        empty: {
          title: "暂无可用模型",
          hint: "请先在模型管理页面添加模型",
        },
        main: {
          label: "主模型",
          placeholder: "选择主模型...",
          hint: "主要用于处理复杂任务，默认使用模型管理页面设置的默认模型",
        },
        smallFast: {
          label: "小型/快速模型（可选）",
          placeholder: "选择快速模型...",
          hint: "可选配置，用于处理简单任务以获得更快的响应速度",
          options: {
            noneLabel: "不使用快速模型",
            noneDescription: "所有任务都使用主模型处理",
            tagFast: " • 快速",
          },
        },
        visual: {
          label: "视觉模型（可选）",
          placeholder: "选择视觉模型...",
          hint: "可选配置，用于图像理解与处理",
          options: {
            noneLabel: "不使用视觉模型",
            noneDescription: "所有任务都使用主模型处理",
            tagVisual: " • 视觉",
          },
        },
        compact: {
          label: "长上下文模型（可选）",
          placeholder: "选择长上下文模型...",
          hint: "可选配置，用于长上下文压缩与处理",
          options: {
            noneLabel: "不使用长上下文模型",
            noneDescription: "所有任务都使用主模型处理",
            tagCompact: " • 长上下文",
          },
        },
      },
      memoryThreshold: {
        label: "记忆相似度阈值",
        subLabel: "阈值",
        strict: "严格",
        loose: "宽松",
        hint: "调整相关记忆的召回阈值。阈值越低匹配越严格；阈值越高召回的记忆越多。",
      },
      buttons: {
        close: "关闭",
      },
    },
    process: {
      execute: "开始执行流程",
    },
    hints: {
      taskMode: "按 Enter 发送 • Shift+Enter 换行 • 支持 Markdown 格式",
      processMode: "填写流程描述后点击执行按钮",
    },
    status: {
      running: "运行中",
      completed: "已完成",
      failed: "失败",
      pending: "等待中",
      paused: "已暂停",
    },
    planDetails: {
      collapsibleTitle: "详细信息",
      badge: {
        memory: "Memory",
        plan: "Plan",
      },
      memory: {
        title: "Memory 信息",
        stats: {
          found: "找到: {count} 条",
          used: "使用: {count} 条",
        },
        enhancedGoalTitle: "增强目标:",
        relatedTitle: "相关记忆:",
        unknownCategory: "未知类别",
      },
      plan: {
        title: "执行计划",
        goalTitle: "目标:",
        stepsTitle: "执行步骤 ({count}):",
        dependenciesPrefix: "依赖: ",
      },
    },
    vibeMode: {
      tabs: {
        task: "Task 模式",
        process: "Process 模式",
      },
      descriptions: {
        task: {
          title: "Task 模式",
          text: "一次性任务模式，快速完成特定目标。适合探索和实验。",
          examplesTitle: "使用举例：",
          examplesText: "完成PPT编写、生成海报、制作网站等",
        },
        process: {
          title: "Process 模式",
          text: "可复用的Agent流程，适合后续Build和Deploy。需要定义流程描述和输入输出样例。",
          examplesTitle: "使用举例：",
          examplesText: "描述客服机器人流程、审计助手等",
          examples: {
            apiExample: {
              title: "Token Refresh 演示",
              labels: {
                currentUser: "当前用户:",
                accessTokenStatus: "Access Token状态:",
                refreshTokenStatus: "Refresh Token状态:",
                wsStatus: "WebSocket状态:",
                wsError: "WebSocket错误:",
              },
              status: {
                obtained: "已获取",
                notObtained: "未获取",
              },
              autoApi: {
                title: "使用自动Token刷新的API调用",
              },
              common: {
                errorPrefix: "错误:",
              },
              models: {
                success: "成功获取 {count} 个模型",
                refetch: "重新获取",
              },
              manualApi: {
                title: "手动API调用（用于对比）",
                button: "手动调用API",
                failed: "API调用失败",
              },
              guide: {
                title: "双Token机制功能说明:",
                accessToken: "Access Token: 30分钟过期，用于API调用",
                refreshToken: "Refresh Token: 7天过期，用于刷新access token",
                autoRefresh: "自动刷新: Access token过期前1分钟自动刷新",
                offlineRecovery: "离线恢复: 离开几小时回来仍可自动恢复登录状态",
                wsSupport: "WebSocket支持: WebSocket连接也支持自动token刷新",
                tokenRotation: "Token轮换: 每次刷新生成新的refresh token，提高安全性",
                seamless: "无感知体验: 用户永远不会因为token过期而需要重新登录",
                smartCache: "智能缓存: 优先检查refresh token过期时间",
              },
            },
          },
        },
      },
      form: {
        processDescription: {
          label: "流程描述",
          placeholder: "详细描述这个Agent流程的用途、适用场景和预期行为...",
        },
        examples: {
          label: "输入输出样例",
          optional: "(可选)",
          add: "添加样例",
          badge: "样例 {index}",
          input: {
            label: "输入:",
            placeholder: "输入样例...",
          },
          output: {
            label: "输出:",
            placeholder: "输出样例...",
          },
        },
      },
    },
    layout: {
      common: {
        notStarted: "未开始",
        inProgress: "进行中",
        unknown: "未知",
      },
      status: {
        pending: "待处理",
        running: "执行中",
        completed: "已完成",
        failed: "失败",
        skipped: "已跳过",
      },
      right: {
        titles: {
          stepDetail: "步骤详情",
          executionLogs: "执行日志",
        },
        labels: {
          resultData: "结果数据",
          stepData: "步骤数据",
          fileOutputs: "文件输出",
          executionTime: "执行时间",
          currentStep: "当前步骤：",
          none: "无",
        },
        buttons: {
          previewAll: "预览全部",
          prevStep: "上一步",
          nextStep: "下一步",
        },
        tooltips: {
          previewAllFiles: "预览所有文件",
          previewFile: "预览文件",
        },
        branch: {
          conditionalNode: "条件分支节点",
          optionalBranches: "可选分支",
          branchCondition: "分支条件",
          requiredBranch: "需要分支",
        },
        counts: {
          files: "{count} 个文件",
          logs: "{count} 条日志",
          fileOutputsWithCount: "文件输出（{count}）",
        },
        empty: {
          selectStepHint: "选择一个步骤以查看详情",
          noLogs: "暂未产生执行日志",
        },
        misc: {
          unknownFile: "未知文件",
        },
      },
      center: {
        titles: {
          dag: "DAG 执行图",
        },
        labels: {
          vertical: "垂直",
          horizontal: "水平",
          layoutVerticalTitle: "垂直布局 (Top-Bottom)",
          layoutHorizontalTitle: "水平布局 (Left-Right)",
          phase: "执行阶段",
          updatedAt: "更新时间：",
          status: "状态：",
          conditionalBranchNode: "条件分支节点",
          optionalBranches: "可选分支",
          branchCondition: "分支条件",
          requiredBranch: "需要分支：",
          branches: "分支：",
          tools: "工具：",
          pureAnalysis: "纯分析",
          description: "描述：",
          result: "执行结果",
        },
        planning: {
          title: "AI 正在制定执行计划",
          subtitle: "智能分析任务并构建最优执行路径",
          steps: {
            analyze: "分析任务需求",
            buildGraph: "构建执行图",
            optimizePath: "优化执行路径",
          },
          spinner: "正在构建执行图...",
        },
        time: {
          title: "时间信息",
          startedAt: "开始时间：",
          completedAt: "完成时间：",
          duration: "执行耗时：",
          inProgress: "进行中",
        },
        errors: {
          taskFailedTitle: "任务执行失败",
          taskFailedDesc: "在执行过程中遇到了错误，请查看左侧消息了解详情",
          checkLeft: "检查左侧消息中的错误信息",
          retryTask: "尝试重新提交任务",
        },
      },
      left: {
        title: "任务执行",
        messageTags: {
          result: "结果",
          file: "文件",
        },
      },
    },
    title: "智能体",
    taskId: "任务 ID: {id}",
    logs: {
      event: {
        labels: {
          start: "开始",
          success: "成功",
          completed: "完成",
          failed: "失败",
          llmCall: "LLM调用",
          llmCompleted: "LLM完成",
          llmFailed: "LLM失败",
          toolCall: "工具调用",
          toolCompleted: "工具完成",
          toolFailed: "工具失败",
          memoryGenerate: "记忆生成",
          memoryStore: "记忆存储",
          memoryQuery: "记忆查询",
          compactStart: "压缩开始",
          compactCompleted: "压缩完成",
          skillSelectStart: "技能选择",
          skillSelectEnd: "技能已选择",
          unknown: "未知操作",
        },
        actions: {
          stepStart: "步骤开始",
          stepCompleted: "步骤完成",
          stepFailed: "步骤失败",
          llmStart: "LLM调用开始",
          llmCompleted: "LLM调用完成",
          llmInfo: "LLM调用信息",
          llmResult: "LLM调用结果",
          llmFailed: "LLM调用失败",
          toolStart: "工具调用开始",
          toolCompleted: "工具调用完成",
          toolFailed: "工具调用失败",
          useTool: "使用工具",
          compactStepErrorLog: "步骤错误（日志记录）",
          reactTaskStart: "ReAct模式任务开始",
          reactTaskCompleted: "ReAct模式任务完成",
          reactStepStart: "ReAct步骤开始",
          reactStepCompleted: "ReAct步骤完成",
          memoryGenerateStart: "记忆生成开始",
          memoryGenerateCompleted: "记忆生成完成",
          memoryStoreStart: "记忆存储开始",
          memoryStoreCompleted: "记忆存储完成",
          memoryQuery: "记忆查询",
          memoryQueryCompleted: "记忆查询完成",
          memoryQueryStart: "记忆查询开始",
          compactStart: "上下文压缩开始",
          compactCompleted: "上下文压缩完成",
          skillSelectStart: "技能选择开始",
          skillSelectEnd: "技能选择完成",
          skillName: "技能名称",
        },
        messages: {
          planStart: "开始生成执行计划: {phase}",
          planEnd: "执行计划生成完成 (计划ID: {planId}, 步骤数量: {stepsCount})",
          taskStart: "开始执行任务 (第{iteration}次迭代)",
          taskEnd: "任务执行完成 (第{iteration}次迭代)",
          taskDesc: "任务描述: {taskPreview}",
          compactStart: "🗜️ 上下文压缩开始",
          compactCompleted: "🗜️ 上下文压缩完成",
          stepFailed: "步骤执行失败: {stepName}",
          stepSkipped: "⏭️ 步骤跳过: {stepName}",
          finalAnswerGenerating: "正在生成最终回答...",
          finalAnswerCompleted: "最终回答生成完成",
          goalCheckStart: "开始目标检查...",
          goalCheck: "目标检查",
          goalAchieved: "已达成",
          goalNotAchieved: "未达成",
          confidence: "(置信度: {percent}%)",
          reason: "原因:",
          reasonLabel: "原因: {goalReason}",
          memoryWillStore: "将存储执行记忆: {memoryReason}",
          taskLLMStart: "开始任务级LLM调用: {taskType}",
          taskLLMCompleted: "任务级LLM调用完成: {taskType}",
          planLLMSending: "💭 计划生成：向大模型发送请求 ({modelName})",
          planLLMResponseCompleted: "计划生成：大模型响应完成 ({modelName})",
          useTool: "🔧 使用工具: {toolName}",
          metaTitle: "元信息:",
          fileOutputsGenerated: "生成了 {count} 个输出文件:",
          previewLabel: "预览",
          executionResultPrefix: "执行结果:",
          errorPrefix: "执行错误:",
          visualUpdate: "可视化更新: {type}",
          reactTaskStart: "🔄 ReAct模式任务开始",
          reactTaskCompleted: "🔄 ReAct模式任务完成",
          reactStepStart: "🔄 ReAct步骤开始: {stepName}",
          reactStepCompleted: "🔄 ReAct步骤完成: {stepName}",
          detailsTitle: "详细信息",
          memoryBadge: "记忆",
          storeSuggestion: "存储建议:",
          worthStoring: "值得存储",
          notWorthStoring: "不值得存储",
          insightsLabel: "洞察生成:",
          storageStatusLabel: "存储状态:",
          notStored: "未存储",
          decisionLabel: "决策:",
          taskLabel: "任务:",
          taskInfoLabel: "任务信息:",
          categoryUnknown: "未知类别",
          stepsCount: "步骤数量: {count}",
          goalTitle: "目标:",
          execStepPrefix: "步骤",
        },
        common: {
          fullData: "完整数据",
          itemsSuffix: "条",
        },
        llm: {
          model: "模型",
          contextMessages: "上下文消息",
          paramsTitle: "调用参数",
          temperature: "Temperature:",
          maxTokens: "Max Tokens:",
          topP: "Top P:",
        },
        tool: {
          tool: "工具",
          paramsCount: "参数数量",
          paramsTitle: "工具参数",
        },
        compact: {
          type: "压缩类型",
          model: "压缩模型",
          originalTokens: "原始Token数",
          threshold: "阈值",
          resultTitle: "压缩结果",
          compactedTokens: "压缩后Token数",
          compressionRatio: "压缩率",
          errorTitle: "压缩错误",
          types: {
            individual_dependency: "单独依赖",
            entire_context: "整体上下文",
          },
        },
        memory: {
          task: "查询任务",
          category: "记忆类别",
          found: "找到记忆",
          used: "使用记忆",
          relatedTitle: "相关记忆",
        },
        skill: {
          task: "任务",
          availableCount: "可用技能数",
          selected: "已选择",
          yes: "是",
          no: "否",
          skillName: "技能名称",
          skill: "个可用技能",
          skills: "个可用技能",
        },
      },
      messagesPreview: {
        title: "上下文消息",
        infoTitle: "上下文信息",
        actions: {
          expand: "展开",
          collapse: "收起",
        },
        labels: {
          role: {
            system: "系统",
            user: "用户",
            assistant: "助手",
          },
          messageIndex: "消息 #{index}",
        },
        parse: {
          error: "解析错误",
          failed: "解析失败",
          errorPrefix: "解析错误:",
        },
      },
    },
    replay: {
      controls: {
        stop: "停止回放",
        pause: "暂停",
        play: "开始回放",
        speed: "播放速度",
        speedSuffix: "x",
      },
    },
  },
  deployments: {
    header: {
      title: "部署",
      description: "管理和监控已部署的 Agents 及其访问端点",
      create: "创建部署",
    },
    dialog: {
      createTitle: "创建部署",
      selectBuild: "选择构建",
      selectPlaceholder: "请选择一个构建...",
      name: "部署名称",
      namePlaceholder: "请输入部署名称...",
      descriptionOptional: "描述（可选）",
      descriptionPlaceholder: "请输入描述（可选）",
      creating: "创建中...",
      formRequired: "请先选择构建并填写部署名称",
    },
    status: {
      running: "运行中",
      stopped: "已停止",
      creating: "创建中",
      error: "错误",
    },
    empty: {
      title: "还没有任何部署",
      description: "创建你的第一个部署以发布 Agent",
    },
    card: {
      executions: "次执行",
      last: "最近：",
      created: "创建：",
      copy: "复制",
      stop: "停止",
      start: "启动",
      details: "详情",
      deleteConfirm: "确定要删除这个部署吗？",
      never: "从未",
    },
    messages: {
      fetchDeploymentsFailed: "加载部署列表失败",
      fetchBuildsFailed: "加载构建列表失败",
      createFailed: "创建部署失败",
      startFailed: "启动部署失败",
      stopFailed: "停止部署失败",
      deleteFailed: "删除部署失败",
      endpointCopied: "端点已复制",
    },
    detail: {
      backTitle: "返回到 Deployments 列表",
      notFound: "未找到部署",
      goBack: "返回到 Deployments",
      info: {
        title: "部署信息",
        status: "状态：",
        type: "类型：",
        executions: "执行次数：",
        lastExecuted: "最近执行：",
        created: "创建时间：",
        build: "构建：",
        viewBuild: "查看构建",
        endpointLabel: "端点 URL：",
      },
      test: {
        title: "测试执行",
        placeholder: "输入测试查询或任务...",
        inputPlaceholder: "请输入 {name}...",
        button: {
          execute: "执行",
          executing: "执行中...",
        },
        resultTitle: "结果：",
        apiUsage: {
          title: "API 使用示例：",
          endpoint: "端点：",
          requestBody: "请求体：",
          example: "示例：",
          yourTaskHere: "your task here",
          yourNameHere: "your {name} here",
        },
      },
      yaml: {
        title: "YAML 配置",
        empty: "暂无 YAML 内容",
      },
      executions: {
        title: "执行历史",
        empty: "暂时还没有执行记录",
        task: "任务：",
        threadId: "线程 ID：",
        output: "输出：",
        timeNodes: "耗时：{seconds}s | 节点：{nodes}",
        error: "错误：",
      },
      misc: {
        na: "N/A",
        deleteExecutionConfirm: "确定要删除这个执行记录吗？",
        errorPrefix: "错误：",
      },
      messages: {
        fetchFailed: "加载部署详情失败",
        fetchExecutionsFailed: "加载执行历史失败",
        executionFailed: "执行失败",
        inputRequired: "请填写 {name} 的值",
      },
    },
  },
  traceEventRenderer: {
    skillSelected: "已选择技能",
    taskExecution: "任务执行",
    unknownModel: "未知模型",
    callLLM: "调用 LLM: {model}",
    llmResponse: "LLM 响应",
    unknownTool: "未知工具",
    executeTool: "执行工具: {tool}",
    toolExecutionFinished: "工具执行完成",
    unknownError: "未知错误",
    executionFailed: "执行失败",
    view: "查看",
    codeAndOutput: "代码与输出",
    args: "参数",
    output: "输出",
    executing: "执行中...",
    errorLabel: "错误: ",
    executionCode: "执行代码:",
    outputResult: "输出结果:",
    toolLabel: "工具: ",
    argumentsLabel: "参数:",
    codeLabel: "代码:",
    outputLabel: "输出:",
    modelLabel: "模型: ",
    reasoningLabel: "推理:",
    noReasoning: "(暂无推理)",
    toolCallsLabel: "工具调用:",
    errorTitle: "错误:"
  },
} as const

export default zh
