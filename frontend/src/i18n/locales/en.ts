const en = {
  common: {
    back: "Back",
    loading: "Loading...",
    save: "Save",
    cancel: "Cancel",
    delete: "Delete",
    confirm: "Confirm",
    deleteConfirm: "Are you sure you want to delete this item?",
    notice: "Notice",
    copy: "Copy",
    copied: "Copied!",
    send: "Send",
    pause: "Pause",
    resume: "Resume",
    executing: "Executing",
    planning: "Formulating Plan",
    taskPaused: "Task Paused",
    noData: "No Data",
    errors: {
      unknown: "Unknown error",
    },
  },
  nav: {
    settings: "Settings",
    dashboard: "Dashboard",
    vibe: "Vibe",
    build: "Agents",
    models: "Models",
    files: "Files",
    memory: "Memory",
    monitoring: "Monitoring",
    knowledgeBase: "Knowledge Base",
    knowledgeBaseDetail: "KB Detail",
    userManagement: "User Management",
    login: "Login",
    register: "Register",
    agentStore: "Agent Store",
    tools: "Tools",
    deploy: "Deploy",
    agentDev: "Agent Development",
    task: "Task",
    history: "All Tasks",
    templates: "Templates",
    sections: {
      agentDevelopment: "Agent Development",
      resources: "Resources",
    },
  },
  chatPage: {
    page: {
      emptyTitle: "Describe the goal, {appName} will handle the rest",
      emptyDescription: "Turn your ideas into working agents automatically. Just describe what you need to get started",
      startWith: "You can start with",
    },
    cards: {
      createPPT: {
        title: "Create PPT",
        description: "Sales reports, slides",
        prompt: "Create a PPT for Q3 E-commerce sales analysis"
      },
      dataAnalysis: {
        title: "Data Analysis",
        description: "Trends, feedback",
        prompt: "Analyze customer feedback data and summarize trends"
      },
      designPoster: {
        title: "Design Poster",
        description: "Social media assets",
        prompt: "Design a poster for the upcoming tech conference"
      },
      automatic: {
        title: "Automatic",
        description: "Custom workflows",
        prompt: "Build an automatic email processing workflow"
      }
    },
    input: {
      placeholder: "Describe your task...",
      hintEnter: "Press Enter to send",
      processing: "Processing",
      actions: {
        config: "Configure Model",
        upload: "Upload",
      },
      noModel: "No Model",
      noModelAlert: "No model configured. Please configure a model first.",
    },
    message: {
      assistantTitle: "AI Assistant",
    },
    reasoning: {
      title: "Reasoning Process",
      stepsCompleted: "{completed}/{total} steps completed",
      thinking: "Thinking",
      initializing: "Initializing reasoning process...",
      defaultStepTitle: "Step {index}",
    },
    executionPlan: {
      title: "Execution Plan",
      tooltip: "Click to view execution plan",
    },
    aiOutput: {
      title: "AI Output",
      subtitle: "Generated response",
      generating: "GeneratingModel Reasoning response",
    },
    tokenUsage: {
      input: "Input tokens",
      output: "Output tokens",
    },
    fileUpload: {
      dropHere: "Drop files here",
      uploadDocs: "Upload documents",
      hintDragClick: "Drag & drop or click to browse",
    },
  },
  branding: {
    subtitle: "AI Agents for Everyone",
    tagline: `Run instantly.
Build when you need.`
  },
  settings: {
    title: "System Settings",
    description: "Manage basic configuration and advanced options",
    language: {
      title: "Language",
      description: "Choose the display language",
    },
    versionControl: {
      title: "Autonomous Planning",
      description: "AI breaks down complex tasks automatically"
    },
    team: {
      title: "Built-in Memory",
      description: "Maintains context across tasks"
    },
    automation: {
      title: "Human in Control",
      description: "Guide and adjust AI execution anytime"
    },
    password: {
      title: "Change Password",
      description: "Update your account password",
      current: "Current Password",
      current_placeholder: "Enter current password",
      new: "New Password",
      new_placeholder: "Enter new password",
      confirm: "Confirm New Password",
      confirm_placeholder: "Re-enter new password",
      submit: "Update Password",
      submitting: "Updating...",
      errors: {
        fill_all: "Please fill in all password fields",
        mismatch: "New password and confirmation do not match",
        too_short: "New password must be at least 6 characters",
        network: "Network error, please try again later",
      },
      success: "Password updated successfully",
      failed: "Password update failed",
    },
    defaultModels: {
      title: "Default Model Settings",
      description: "Set default models for different task types. These settings apply only to you.",
      types: {
        general: { title: "General Model", description: "Suitable for most tasks" },
        small_fast: { title: "Small & Fast Model", description: "Lightweight and quick responses" },
        visual: { title: "Vision Model", description: "Understands and processes images" },
        compact: { title: "Long Context Model", description: "Handles long contexts and compression" },
        embedding: { title: "Embedding Model", description: "Generates vector embeddings" },
      },
      actions: {
        clearDefault: "Clear Default",
      },
      labels: {
        selectDefault: "Select Default Model",
        selectModel: "Select a model",
      },
      empty: {
        noModels: "No available models. Please add models on the Models page first.",
      },
      messages: {
        loadFailed: "Failed to load model configuration",
        updated: "Updated: {type}",
        setFailed: "Failed to set default model",
        removed: "Cleared: {type}",
        removeFailed: "Failed to clear default model",
      },
      guide: {
        title: "Usage Guide",
        items: {
          personalOnly: "These settings only apply to you",
          fallbackFirstAvailable: "If no default is set, the system will use the first available model",
          addOnModelsPage: "Add models on the Models management page before selecting",
          adminShared: "Administrators can configure shared models for team use",
        },
      },
    },
  },
  register: {
    mobile_title: "Create Account",
    title: "Register {appName}",
    description: "Create your account to start using the agent platform",
    form: {
      username: "Username",
      username_placeholder: "Enter username",
      password: "Password",
      password_placeholder: "Enter password (at least 6 characters)",
      confirm_password: "Confirm Password",
      confirm_password_placeholder: "Re-enter password",
      submit: "Create Account",
      submitting: "Registering...",
    },
    alerts: {
      success: "Registration successful! Please log in",
      failed: "Registration failed",
      failed_retry: "Registration failed, please try again later",
      password_mismatch: "Passwords do not match",
      password_too_short: "Password must be at least 6 characters",
    },
    login_hint: {
      has_account: "Already have an account?",
      login_now: "Log in now",
    },
    status: {
      agent_running: "Agent service running",
      open_register: "Registration open",
    },
    features: {
      vbd: {
        title: "Autonomous Planning",
        description: "AI that plans before it acts. Breaks down complex tasks and executes step by step.",
      },
      hitl: {
        title: "Built-in Memory",
        description: "Remembers what matters. Keeps context across tasks and conversations.",
      },
      timetravel: {
        title: "Human in Control",
        description: "You guide. AI executes. Review, adjust, and step in anytime.",
      },
    },
  },
  setup: {
    title: "Initialize {appName}",
    description: "Create the first administrator account to complete setup.",
    features: {
      bootstrap: {
        title: "Guided bootstrap",
        description: "Finish first-time initialization in one step.",
      },
      config: {
        title: "Ready to configure",
        description: "After setup, configure models and tools for your team.",
      },
      security: {
        title: "Admin-first security",
        description: "Create a secure administrator account before opening access.",
      },
    },
    form: {
      username: "Admin username",
      password: "Admin password",
      confirmPassword: "Confirm password",
      submit: "Create administrator",
      submitting: "Creating...",
    },
    errors: {
      required: "Please fill in username and password",
      passwordTooShort: "Password must be at least 6 characters",
      passwordMismatch: "Passwords do not match",
      failed: "Setup failed",
    },
  },
  login: {
    mobile_title: "Welcome back",
    title: "Login {appName}",
    description: "Please enter your account information to access the agent platform",
    form: {
      username: "Username",
      username_placeholder: "Enter username",
      password: "Password",
      password_placeholder: "Enter password",
      submit: "Log In",
      submitting: "Logging in...",
    },
    alerts: {
      auth_failed: "Incorrect username or password",
      network_failed: "Login failed, please try again later",
    },
    options: {
       remember_me: "Remember me",
       forgot_password: "Forgot password?",
     },
    register_prompt: "Don't have an account?",
    register_link: "Register now",
    register_closed: "Registration is currently disabled",
     features: {
       version_control: {
         title: "Autonomous Planning",
         description: "AI that plans before it acts. Breaks down complex tasks and executes step by step.",
       },
       team: {
         title: "Built-in Memory",
         description: "Remembers what matters. Keeps context across tasks and conversations.",
       },
       automation: {
         title: "Human in Control",
         description: "You guide. AI executes. Review, adjust, and step in anytime.",
       },
     },
   },
  vibe: {
    title: "Vibe Stage",
    description: "Turn prompts into runnable agents; let LLMs plan automatically and unleash imagination",
    actions: {
      you_can: "You can",
      new_task: "New Task",
      delete_confirm: "Are you sure you want to delete this task?",
    },
    quick_examples: {
      make_ppt: {
        title: "Create PPT",
        description: "Quarterly review, marketing proposal",
      },
      data_analysis: {
        title: "Data Analysis",
        description: "E-commerce sales, customer feedback",
      },
      design_poster: {
        title: "Design Poster",
        description: "Event promotion, social media",
      },
      automation: {
        title: "Automation",
        description: "Email processing, document management",
      },
      more: {
        title: "More",
        description: "Explore endless possibilities",
      },
    },
    list: {
      title: "Task List",
      search_placeholder: "Search tasks...",
      empty: "No task records",
      pagination: {
        summary: "Showing {from}-{to} of {total} tasks",
        prev: "Previous",
        next: "Next",
      },
      status: {
        completed: "Completed",
        running: "Running",
        failed: "Failed",
        pending: "Pending",
        paused: "Paused",
      },
      mode: {
        task: "Task",
        process: "Process",
      },
    },
  },
  userManagement: {
    title: "User Management",
    description: "Manage all user accounts in the system",
    stats: {
      total_users_prefix: "Total ",
      total_users_suffix: " users",
    },
    list: {
      title: "User List",
      description: "View and manage all users",
      search_placeholder: "Search username...",
      table: {
        id: "ID",
        username: "Username",
        role: "Role",
        created_at: "Created At",
        updated_at: "Last Updated",
        actions: "Actions",
        admin: "Admin",
        normal: "User",
        delete: "Delete",
        delete_confirm_title: "Confirm Delete User",
        delete_confirm_description_prefix: "Are you sure you want to delete user \"",
        delete_confirm_description_suffix: "\"? This action cannot be undone and all user data will be permanently removed.",
        cancel: "Cancel",
        confirm_delete: "Delete",
      },
      loading: {
        title: "Loading...",
        description: "Verifying user permissions",
      },
      no_permission: {
        title: "Insufficient Permissions",
        description: "Only administrators can access the user management page",
      },
      alerts: {
        delete_success_prefix: "Deleted successfully: user ",
        delete_success_suffix: " has been removed",
        delete_failed_prefix: "Delete failed: ",
        delete_failed_suffix: "",
      },
      pagination: {
        summary: "Showing {from}-{to} of {total} users",
        prev: "Previous",
        next: "Next",
        page: "Page {current} of {total}",
      },
    },
  },
  dashboard: {
    stats: {
      total_tasks: "Total Tasks",
      active_agents: "Active Agents",
      deployed_apps: "Deployed Apps",
      today_calls: "Calls Today",
    },
    features: {
      title: "Main Features",
      vibe: {
        title: "VIBE",
        description: "Create intelligent agents, auto-planning to unleash imagination",
      },
      build: {
        title: "BUILD",
        description: "Build agent workflows into deployable applications",
      },
      deploy: {
        title: "DEPLOY",
        description: "Deploy applications to production to deliver business value",
      },
      models: {
        title: "Model Management",
        description: "Configure and manage AI models, set default models",
      },
      monitoring: {
        title: "Monitoring",
        description: "Monitor LLM calls, tool usage, and system performance",
      },
      files: {
        title: "File Management",
        description: "Upload, manage, and view project files",
      },
    },
    tasks: {
      title: "Recent Tasks",
      search_placeholder: "Search tasks...",
      new_task: "New Task",
      empty: "No task records",
      delete_confirm: "Are you sure you want to delete this task?",
      status: {
        completed: "Completed",
        running: "Running",
        failed: "Failed",
        pending: "Pending",
      },
    },
    files: {
      title: "Recent Files",
      view_all: "View All",
      empty: "No files",
    },
  },
  monitoring: {
    title: "Monitoring",
    description: "Monitor platform performance, LLM calls, and tool usage",
    cards: {
      totalCalls: "Total Calls",
      successRate: "Success Rate",
      avgResponseTime: "Avg Response Time",
      todayCalls: "Calls Today"
    },
    models: {
      title: "Model Statistics",
      empty: "No model data",
      summary: "Success Rate: {success}% • Total Tasks: {total}",
      usageRate: "Usage Rate"
    },
    tools: {
      title: "Tool Usage",
      empty: "No tool usage data",
      avgDuration: "Average duration: {duration}s",
      usageCount: "{count} uses"
    }
  },
  tools: {
    header: {
      title: "Tool Management",
      description: "Manage built-in tools and MCP server configurations",
    },
    tabs: {
      all: "All Tools",
      mcp: "MCP Servers",
    },
    stats: {
      mcpServers: {
        title: "MCP Servers",
        activeCount: "{count} active servers",
      },
      builtinTools: {
        title: "Built-in Tools",
        summary: "{total} tools ({enabled} enabled)",
      },
      totalTools: {
        title: "Total Tools",
        summary: "{total} tools ({enabled} enabled)",
      },
    },
    list: {
      title: "Tool List",
      description: "View and manage all available tools",
      searchPlaceholder: "Search tools...",
      disabledBadge: "Disabled",
      sourcePrefix: "Source: {source}",
      actions: {
        enable: "Enable",
        disable: "Disable",
      },
      empty: {
        title: "No tools found",
        description: "Try adjusting your search or add a new tool",
      },
      noDescription: "No description provided",
      transport: "Transport",
      usedByAgents: "Used by {count} agents",
    },
    mcp: {
      title: "MCP Server Management",
      description: "Manage your Model Context Protocol (MCP) server configurations",
      addServer: "Add Server",
      badge: "MCP Server",
      dialog: {
        addTitle: "Add MCP Server",
        editTitle: "Edit MCP Server",
        description: "Configure MCP server connection parameters",
      },
      form: {
        nameLabel: "Server Name *",
        namePlaceholder: "Enter server name",
        transportLabel: "Transport *",
        transportPlaceholder: "Select transport",
        descriptionLabel: "Description",
        descriptionPlaceholder: "Enter server description",
        fieldPlaceholderPrefix: "Please enter {field}",
      },
      transports: {
        stdio: { label: "Standard I/O", description: "Communicate via standard input/output streams" },
        sse: { label: "Server-Sent Events", description: "Communicate via Server-Sent Events" },
        websocket: { label: "WebSocket", description: "Communicate via WebSocket protocol" },
        streamable_http: { label: "Streamable HTTP", description: "Communicate via HTTP streaming" },
      },
      buttons: {
        cancel: "Cancel",
        update: "Update",
        create: "Create",
        save: "Save",
      },
      empty: {
        title: "No MCP server configuration",
        description: "Click \"Add Server\" to start configuration",
      },
      list: {
        defaultBadge: "Default",
        createdAt: "Created At: {date}",
      },
      alerts: {
        nameRequired: "Server name cannot be empty",
        saveFailed: "Save failed",
        deleteFailed: "Delete failed",
        toggleFailed: "Toggle status failed",
      },
      delete: {
        confirm: "Are you sure to delete server \"{name}\"?",
      },
    },
    categories: {
      development: "Development Tools",
      search: "Search Tools",
      knowledge: "Knowledge Base",
      file: "File Operations",
      utility: "Utility Tools",
      creative: "Creative Tools",
      analysis: "Analysis Tools",
      mcp: "MCP Tools",
    },
    badges: {
      types: {
        tool: "Tool",
        basic: "Basic",
        file: "File",
        knowledge: "Knowledge",
        image: "Image",
        vision: "Vision",
        special_image: "Image",
      },
      source: {
        builtin: "Built-in",
      },
      basic: "Basic",
      file: "File",
      knowledge: "Knowledge",
      image: "Image",
      vision: "Vision",
      special_image: "Image",
      builtin: "Built-in",
    },
  },
  files: {
    header: {
      title: "File Management",
      description: "Manage project files: upload, preview, download, and delete.",
    },
    search: {
      placeholder: "Search files...",
    },
    actions: {
      upload: "Upload",
      uploading: "Uploading...",
      delete: "Delete",
      download: "Download",
      preview: "Preview",
    },
    sidebar: {
      folders: "Folders",
      allFiles: "All Files",
      agents: "Agents",
      noAgents: "No agents found",
      system: "System",
      userUploads: "User Uploads",
    },
    breadcrumb: {
      files: "Files",
      unknownAgent: "Unknown Agent",
      unknownCategory: "Unknown Category",
    },
    selection: {
      selected: "{count} selected",
    },
    table: {
      name: "Name",
      type: "Type",
      size: "Size",
      dateModified: "Date Modified",
      empty: {
        loading: "Loading files...",
        noMatch: "No files match your search.",
        noFiles: "No files found.",
      },
    },
    time: {
      justNow: "Just now",
      minsAgo: "{count} mins ago",
      hoursAgo: "{count} hours ago",
      daysAgo: "{count} days ago",
      monthsAgo: "{count} months ago",
      yearsAgo: "{count} years ago",
    },
    delete: {
      confirmSingle: "Are you sure you want to delete {name}?",
      confirmMultiple: "Are you sure you want to delete {count} files?",
    },
    previewDialog: {
      buttons: {
        download: "Download",
        openInNewWindow: "Open in New Window",
      },
      loading: "Loading file content...",
      imageError: {
        title: "Image failed to load",
        hint: "Try downloading to view",
      },
      emptyContent: "File content is empty",
      errors: {
        loadFailed: "Failed to load file",
        docxRenderFailed: "Failed to render DOCX preview",
        cors: "CORS error: Unable to access the file. This might be a browser cache issue, please try refreshing the page.",
        networkErrorWithMsg: "Network error: {msg}",
      },
    },
  },
  memory: {
    loading: "Loading memories...",
    header: {
      backTitle: "Back to Home",
      title: "Memory Center",
      description: "Store, search, and manage agent memory items",
      create: "New Memory",
    },
    createDialog: {
      title: "Create Memory",
      description: "Fill in the content, keywords, tags, and metadata to add a new memory",
    },
    stats: {
      total: "Total Memories",
      categories: "Categories",
      tags: "Tags",
      storeType: "Store Type",
    },
    filters: {
      title: "Filters",
      category: {
        label: "Category",
        all: "All",
        placeholder: "Select category",
      },
      categoryOptions: {
        general: "General",
        react_memory: "React Memory",
        dag_plan_execute_memory: "DAG Plan Execute Memory",
      },
      search: {
        label: "Search",
        placeholder: "Search content, keywords, or tags...",
      },
      similarity: {
        label: "Similarity Threshold",
        strict: "Strict",
        lenient: "Lenient",
      },
      date: {
        fromLabel: "From",
        toLabel: "To",
        fromPlaceholder: "Start date",
        toPlaceholder: "End date",
      },
    },
    item: {
      keywordPrefix: "Keyword",
      tagPrefix: "Tag",
    },
    list: {
      empty: "No memories found",
    },
    viewDialog: {
      title: "View Memory",
      description: "Detailed information of the selected memory item",
      labels: {
        category: "Category",
        time: "Time",
        content: "Content",
        keywords: "Keywords",
        tags: "Tags",
        metadata: "Metadata",
      },
      close: "Close",
      edit: "Edit",
    },
    editDialog: {
      title: "Edit Memory",
      description: "Modify the content, keywords, tags, or metadata",
    },
    deleteDialog: {
      title: "Delete Memory",
      description: "Are you sure you want to delete this memory? This action cannot be undone.",
      bulkDescription: "Are you sure you want to delete {count} memories? This action cannot be undone.",
      delete: "Delete",
    },
    form: {
      contentLabel: "Content",
      contentPlaceholder: "Enter memory content...",
      keywordsLabel: "Keywords",
      keywordsPlaceholder: "Enter keywords, separated by commas",
      tagsLabel: "Tags",
      tagsPlaceholder: "Enter tags, separated by commas",
      metadataLabel: "Metadata (JSON)",
      metadataPlaceholder: "Enter metadata in JSON format, e.g. {\"source\":\"web\"}",
    },
    sidebar: {
      contextSource: "Context Source",
      allMemories: "All Memories",
      byCategory: "By Category",
    },
    actions: {
      exportSelected: "Export Selected ({count})",
      exportCsv: "Export CSV",
      deleteSelected: "Delete Selected ({count})",
      clearFilter: "Clear Filter",
    },
    empty: {
      title: "No memories found",
      searchHint: "Try adjusting your search query.",
      categoryHint: "There are no memories in this category yet.",
    },
    errors: {
      deleteFailed: "Failed to delete memory",
      deleteBatchFailed: "Failed to delete memories",
      createFailed: "Failed to create memory",
      updateFailed: "Failed to update memory",
      unknown: "Unknown error",
    },
  },
  templates: {
    title: "Templates",
    subtitle: "Jumpstart your workflow with pre-built agents.",
    searchPlaceholder: "Search templates...",
    useTemplate: "Use Template",
    noResults: "No templates found matching your criteria.",
    categoryTitles: {
      featured: "Featured Templates",
      all: "All",
      healthcare_fitness: "Healthcare & Fitness",
      general_productivity: "General & Productivity",
      customer_service: "Customer Service",
      finance_lms_ops: "Finance, LMS & Ops",
      security: "Security"
    },
    sections: {
      knowledge: "Knowledge & Research",
      sales: "Sales & Outreach"
    },
    items: {
      internal_qa: {
        title: "Internal Q&A",
        description: "Connect to internal docs and answer employee questions automatically."
      },
      deep_research: {
        title: "Deep Research",
        description: "Research companies, web, and reports to compile comprehensive data."
      },
      outbound_bdr: {
        title: "Outbound BDR",
        description: "Scrape leads and draft personalized outreach emails."
      },
      inbound_lead: {
        title: "Inbound Lead Processing",
        description: "Qualify incoming leads and auto-schedule meetings."
      },
      crm_enrichment: {
        title: "CRM Enrichment",
        description: "Enrich CRM contacts with data from the web and LinkedIn."
      },
      sales_call_notes: {
        title: "Sales Call Notes",
        description: "Transcribe and summarize sales calls for follow-ups."
      },
      invoice_automation: {
        title: "Invoice Automation",
        description: "Extract data from invoices and update accounting systems."
      }
    }
  },
  sidebar: {
    tasks: {
      empty: "No tasks",
      viewAll: "View All",
    },
    user: {
      loggedIn: "Logged in",
      notLoggedIn: "Not logged in",
      logoutTitle: "Logout",
      defaultName: "User",
    },
  },
  agentStore: {
    title: "Agent Store",
    subtitle: "Discover and use professional AI Agents across data analysis, code generation, text processing, and more",
    builtIn: {
      title: "Built-in Agents",
      badgeOfficial: "Official",
    },
    text2sql: {
      title: "Text2SQL Assistant",
      descriptionShort: "Intelligent database query assistant that turns natural language into SQL",
      featuresParagraph: "Supports multiple database types with schema analysis, SQL optimization, and result formatting. Ideal for data analysis, reporting, and business queries.",
      startUsing: "Start Using",
      config: {
        subtitle: "Select a database connection to start querying",
        connectionsTitle: "Database Connections",
        startQuery: "Start Query",
        noConnections: "No database connections yet. Add a new database below.",
        readonlyBadge: "Read-only",
        addNewDatabaseTitle: "Add New Database",
        addDatabase: "Add Database",
        adding: "Adding...",
        form: {
          nameLabel: "Database Name",
          namePlaceholder: "Enter a database name, e.g., Production DB",
          typeLabel: "Database Type",
          urlLabel: "Connection String",
          urlPlaceholder: "Enter the database connection string",
          readonlyLabel: "Connect in read-only mode (recommended)",
        },
        alerts: {
          fillAll: "Please fill in all required information",
          addFailedPrefix: "Failed to add database: ",
          unknownError: "Unknown error",
          addFailed: "Failed to add database, please try again later",
          testSuccess: "Connection test successful: {message}",
          testFailedPrefix: "Connection test failed: ",
          testFailed: "Connection test failed, please try again later",
        },
        statusBadge: {
          connected: "Connected",
          disconnected: "Disconnected",
          error: "Error",
        },
        loginRequiredTitle: "Please log in",
        loginRequiredDescription: "You need to log in to use the Text2SQL Assistant",
      },
      chat: {
        login: {
          title: "Please log in",
          description: "Login required to use the Text2SQL Agent",
        },
        database: {
          unknown: "Unknown Database",
        },
        connection: {
          connected: "Connected",
          connecting: "Connecting",
          initializing: "Initializing connection...",
          wsConnecting: "Establishing WebSocket connection...",
          ready: "Connection ready",
        },
        history: {
          title: "History",
          newConversation: "Start new conversation",
          emptyTitle: "No history",
          emptyDescription: "Create a new task to start querying the database",
          delete: "Delete task",
          deleteConfirm: "Are you sure you want to delete this task?",
          deleteFailed: "Failed to delete task",
        },
        mode: {
          readOnly: "Read-only Mode",
          readWrite: "Read-Write Mode",
        },
        initializing: "Initializing Text2SQL Assistant...",
        welcome: {
          title: "Welcome to Text2SQL Assistant",
          description: "Select a conversation from the left, or type below to start querying",
          featuresTitle: "Text2SQL Assistant can help you:",
          features: {
            naturalToSql: "Convert natural language to SQL queries",
            analyzeSchema: "Analyze database schema",
            executeAndShow: "Execute SQL and display results",
            optimizeQuery: "Provide query optimization suggestions",
          },
        },
        input: {
          title: "Enter your query request",
          hint: "Tip: The assistant will analyze the database schema and generate SQL automatically",
          placeholder: "Please enter your query...",
        },
        status: {
          ready: "Ready",
          connecting: "Connecting...",
          initializing: "Initializing...",
        },
        aiSummary: {
          title: "AI Summary",
        },
        table: {
          title: "Query Results",
          titleWithCount: "Query Results ({count} records)",
          buttons: {
            chart: "Charts",
            bar: "Bar Chart",
            pie: "Pie Chart",
            line: "Line Chart",
            copyData: "Copy Data",
            exportCsv: "Export CSV",
            copy: "Copy",
          },
        },
        steps: {
          analyzeSchema: "Analyze Database Schema",
          executeSql: "Execute SQL Query",
          formatResult: "Format Result",
          defaultStep: "Query Processing Step",
          plan: "Planning Strategy",
        },
        thinking: {
          schemaCompleted: "Database schema analysis completed",
          schemaRunning: "Analyzing database schema and relationships...",
          sqlGenCompleted: "SQL query generation completed",
          sqlGenRunning: "Generating and optimizing SQL query...",
          executeCompleted: "SQL query execution completed",
          executeRunning: "Executing SQL query...",
          formatCompleted: "Query result formatting completed",
          formatRunning: "Organizing and formatting query results...",
          planCompleted: "Query strategy planning completed",
          planRunning: "Planning query strategy and execution plan...",
          analysisCompleted: "Problem analysis completed",
          analysisRunning: "Analyzing the problem and database structure...",
          sqlStatementCompleted: "SQL statement generation completed",
          sqlStatementRunning: "Generating SQL query statements...",
          queryCompleted: "Query execution completed",
          queryRunning: "Executing query operation...",
          stepCompleted: "Query step processing completed",
          stepRunning: "Processing query step...",
        },
      },
      timeline: {
        title: "Thinking Timeline",
        stepType: {
          planning: "Planning",
          analysis: "Analysis",
          sql_generation: "SQL Generation",
          execution: "Execution",
          result: "Result"
        },
        status: {
          pending: "Pending",
          running: "Running",
          completed: "Completed",
          failed: "Failed"
        },
        details: {
          sql: "Generated SQL",
          result: "Query Result",
          error: "Error"
        },
        actions: {
          cancelAutoCollapse: "Cancel Auto Collapse",
          toggleStep: "Toggle Step: {name}"
        },
        labels: {
          dependencies: "Dependencies: {count}",
          unknownStatus: "Unknown Status",
          basedOnSteps: "Based on steps: {steps}"
        }
      },
      charts: {
        noNumericData: "No numeric data suitable for charts",
        switchToTable: "Switch to Table",
        bar: {
          rankTop: "{valueColumn} Ranking (Top {top})",
        },
        pie: {
          distributionTop: "{valueColumn} Distribution (Top {top})",
        },
        line: {
          trendTop: "{valueColumn} Trend (Top {top})",
        },
        predict: {
          buttonPredict: "AI Predict",
          buttonPredicting: "Predicting...",
          successDefault: "Prediction data generated successfully",
          failedDefault: "Prediction generation failed",
          requestFailedHttp: "Prediction request failed: HTTP {status}",
          requestFailedNetwork: "Prediction request failed: Unable to connect to prediction service",
        },
        analysis: {
          trendAnalysisLabel: "Trend Analysis:",
          confidence: "Confidence: {confidence}",
        },
        legend: {
          historical: "Historical Data",
          prediction: "Predicted Data",
        },
      },
    },
    stats: {
      users: "{count} users",
      usesToday: "{count} uses today",
    },
    badges: {
      smartCorrection: "Smart Correction",
    },
    moreBuiltIn: {
      title: "More Built-in Agents",
      comingSoon: "Coming soon...",
      description: "We are building more professional AI Agents, including code assistants, data analysis, and document processing.",
    },
    deployments: {
      title: "VIBD Deployed Agents",
      badgeCommunity: "Community",
      noneTitle: "No deployed agents",
      noneDescription: "Once you deploy custom agents via the VIBE → BUILD → DEPLOY flow, they will appear here.",
      helperText: "Use our VIBE system to build and deploy your own AI Agents",
    },
    actions: {
      createAgent: "Create Agent",
    },
    footer: {
      title: "Get Started with AI Agents",
      description: "Choose the right agent for your needs and start intelligent workflows immediately",
      tryText2SQL: "Try Text2SQL",
      createCustomAgent: "Create Custom Agent",
    },
  },
  models: {
    header: {
      title: "Model Management",
      description: "Configure and manage AI models, set default models",
      add: "Add Model",
    },
    search: {
      placeholder: "Search models...",
    },
    tabs: {
      llm: "LLM Models",
      embedding: "Embedding Models",
      image: "Image Generation Models",
    },
    section: {
      enabledModels: "Enabled Models",
      configuredCount: "{count} configured",
      exploreProviders: "Model Providers",
    },
    deleteConfirm: "Are you sure you want to delete this model?",
    deleteTitle: "Delete Model",
    errors: {
      setDefaultFailed: "Failed to set default model",
      removeDefaultFailed: "Failed to remove default model",
    },
    defaults: {
      general: "Default Model",
      fast: "Fast Model",
      visual: "Visual Model",
      compact: "Long Context Model",
      embedding: "Embedding Model",
      image: "Image Generation Model",
      image_edit: "Image Edit Model",
      shared: "Shared",
    },
    card: {
      owner: "Owner",
      shared: "Shared",
      inactive: "Inactive",
      fields: {
        provider: "Provider",
        name: "Name",
        temperature: "Temperature",
        not_set: "Not set",
        dimension: "Dimension",
        base_url: "Base URL",
        default: "Default",
        abilities: "Abilities",
        description: "Description",
        created: "Created",
        updated: "Updated",
        unknown: "Unknown",
      },
      actions: {
        edit: "Edit",
        setDefault: "Set Default",
        connect: "Connect",
      },
      status: {
        connected: "Connected",
      },
    },
    empty: {
      llm: "No configured LLM models",
      embedding: "No configured embedding models",
      image: "No configured image models",
      description: "Click 'Add Model' to start configuration",
    },
    dialog: {
      connectProviderTitle: "Connect {provider}",
      connectProviderDescription: "Enter your API Key to fetch available models.",
      fetchingModels: "Fetching models...",
      modelsFetchedAfterConnect: "Models will be fetched after connecting.",
      cancel: "Cancel",
      add: "Create Model",
      editTitle: "Edit Model",
      createTitle: "Create Model",
      editDescription: "Modify the configuration of an existing model",
      createDescription: "Configure a new model to use in the system",
      providerModelsTitle: "{provider} Models",
      manageDescription: "Manage your configured models for this provider.",
      noModelsConfigured: "No models configured.",
      availableModels: "Available Models",
    },
    form: {
      modelId: "Model ID",
      category: "Category",
      provider: "Provider",
      name: "Model Name",
      temperature: "Temperature",
      dimension: "Vector Dimension",
      apiKey: "API Key",
      apiKeyPlaceholder: "sk-...",
      apiKeyPlaceholderEdit: "Leave blank to keep unchanged",
      baseUrl: "Base URL",
      baseUrlPlaceholder: "https://api.openai.com/v1",
      abilities: "Abilities",
      description: "Description",
      setDefault: "Set as default model",
      defaultPlaceholder: "Select default type...",
      shareWithUsers: "Share this model with all users",
      abilitiesPlaceholder: "Select abilities...",
      update: "Update Model",
      create: "Create Model",
    },
    abilities: {
      chat: "Chat",
      vision: "Vision",
      tool_calling: "Tool Calling",
      thinking_mode: "Thinking Mode",
      embedding: "Embedding",
      generate: "Generate",
      edit: "Edit",
    },
    defaultDialog: {
      title: "Set Default Model",
      description: "Set the default model type for {modelId}",
      selectType: "Select model type",
      current: "Current",
      removeTitle: "Remove default",
      cancel: {
        general: "Cancel Default",
        fast: "Cancel Fast",
        visual: "Cancel Visual",
        compact: "Cancel Long Context",
        embedding: "Cancel Embedding",
      },
      close: "Close",
    },
    providers: {
      openai: {
        description: "Access GPT-4o, GPT-4 Turbo, and DALL-E 3 models. Industry standard for reasoning and creativity.",
      },
      azure_openai: {
        description: "Enterprise-grade OpenAI models hosted on Azure.",
      },
      zhipu: {
        description: "GLM-4 models optimized for Chinese and English tasks. Strong reasoning capabilities.",
      },
      dashscope: {
        description: "Qwen models from Alibaba Cloud. Strong performance in Chinese language tasks.",
      },
      gemini: {
        description: "Multimodal capabilities natively built-in. Excellent for analyzing images, video, and large text corpuses.",
      },
      claude: {
        description: "Claude 3.5 Sonnet and Opus models. Known for high steerability, large context windows, and coding tasks.",
      },
      xinference: {
        description: "Local inference platform for running open-source LLMs, embedding models, and image models. Deploy models on your own infrastructure.",
      },
      openai_embedding: {
        description: "Dedicated embedding models from OpenAI.",
      },
    },
  },
  kb: {
    errors: {
      loadFailed: "Failed to load knowledge base",
      nameRequired: "Knowledge base name is required",
      createFailed: "Failed to create knowledge base",
      uploadFileRequired: "Please select a file to upload",
      uploadFailedFile: "Failed to upload file: {name}",
      uploadFailed: "Upload failed",
      startUrlRequired: "Start URL is required",
      webIngestFailed: "Website import failed",
      deleteFailed: "Failed to delete knowledge base: {name}",
      deleteFailedGeneric: "Failed to delete knowledge base",
      unsupportedFileType: "Unsupported file type",
    },
    actions: {
      deleteConfirm: "Are you sure you want to delete knowledge base {name}? This action cannot be undone.",
    },
    loading: {
      loadingKB: "Loading knowledge base...",
    },
    header: {
      title: "Knowledge Base Management",
      description: "Manage document collections and search indices",
      new: "New Knowledge Base",
    },
    search: {
      placeholder: "Search knowledge base...",
    },
    card: {
      documentsLabel: "Documents",
      chunksLabel: "Text Chunks",
      embeddingsLabel: "Vectors",
      parsesLabel: "Parses",
      noDescription: "No description",
      actions: {
        viewDetail: "View Details",
      },
      status: {
        active: "Active",
      },
    },
    empty: {
      noKB: "No knowledge base",
      searchNoMatch: "No knowledge base matches your search",
      hintSearch: "Try other search keywords",
      hintCreate: "Click 'New Knowledge Base' to create your first knowledge base",
    },
    dialog: {
      createTitle: "New Knowledge Base",
      createDescription: "Create a new knowledge base and upload documents or import from website",
      basicInfo: {
        title: "Basic Information",
        nameLabel: "Knowledge Base Name",
        namePlaceholder: "Leave blank to use filename or URL as the name",
        descriptionLabel: "Description",
        descriptionPlaceholder: "Knowledge base description (optional)",
      },
      tabs: {
        file: "File Upload",
        web: "Website Import",
      },
      fileUpload: {
        title: "Document Upload",
        dropOrClick: "Drag files here or click to select",
        supportedFormats: "Supported formats: PDF, TXT, HTML, MD, Word, Excel, PPT, etc.",
        selectFiles: "Select Files",
        selectedTitle: "Selected Files",
        remove: "Remove",
        uploading: "Uploading...",
        uploadSuccess: "Upload Successful",
        uploadFailed: "Upload Failed",
        progressTitle: "Upload Progress",
        processing: "Processing...",
        processSuccess: "Processing Success",
        processResult: {
          createDocuments: "Create Documents",
          textChunks: "Text Chunks",
          vectors: "Vectors",
        },
      },
      webImport: {
        title: "Website Import",
        description: "Respect robots.txt and crawl target website content, then index",
        basic: {
          title: "Basic Configuration",
          startUrl: "Start URL",
          maxPages: "Max Pages",
          crawlDepth: "Crawl Depth",
          concurrentRequests: "Concurrent Requests",
        },
        advanced: {
          title: "Advanced Configuration",
          urlPatterns: "URL Patterns",
          excludePatterns: "Exclude Patterns",
          contentSelector: "Content Selector",
          removeSelectors: "Remove Selectors",
          requestDelay: "Request Delay (ms)",
          requestDelaySeconds: "Request Delay (seconds)",
          timeout: "Request Timeout (ms)",
          timeoutSeconds: "Request Timeout (seconds)",
          sameDomainOnly: "Same Domain Only",
          respectRobotsTxt: "Respect robots.txt",
          hintMultiple: "Separate multiple patterns with commas",
          hintContentSelector: "CSS selector for main content",
        },
        status: {
          progressTitle: "Crawl Progress",
          crawling: "Crawling website, this may take a few minutes...",
          success: "Crawl Successful",
          done: "Crawl Completed",
        },
        result: {
          pages: "Crawled Pages",
          viewWarnings: "View Warnings",
        },
      },
    },
    index: {
      title: "Index Configuration",
      parseMethod: "Parse Method",
      parseOptions: {
        default: "Default",
        pypdf: "PyPDF",
        pdfplumber: "PDFPlumber",
        unstructured: "Unstructured",
        pymupdf: "PyMuPDF",
        deepdoc: "DeepDoc",
      },
      chunkStrategy: "Chunking Strategy",
      chunkOptions: {
        recursive: "Recursive",
        fixed_size: "Fixed Size",
        markdown: "Markdown",
      },
      chunkSize: "Chunk Size",
      chunkOverlap: "Chunk Overlap",
      embeddingModelId: "Embedding Model",
      embeddingBatchSize: "Batch Size",
      startImport: "Start Import",
      saveConfig: "Save Configuration",
    },
    detail: {
      viewingDetails: "Viewing details for {name}",
      loadingDetail: "Loading knowledge base detail...",
      notFound: "Knowledge base not found",
      backToList: "Back to Knowledge Bases",
      tabs: {
        files: "Files",
        search: "Search Test",
        settings: "Settings",
      },
      files: {
        title: "Files",
        addSource: "Add Source",
        addDialogTitle: "Add Data Source",
        addDialogDescription: "Choose a data source to add to this knowledge base",
        uploading: "Uploading...",
        upload: "Upload",
      },
      uploaded: {
        title: "Uploaded Files",
        indexed: "Indexed",
        emptyTitle: "No files",
        emptyHint: "Uploaded files will appear here",
        delete: "Delete document",
        confirmDelete: "Are you sure you want to delete this document?",
      },
      process: {
        title: "Processing Results",
        labels: {
          file: "File",
          document: "Document",
          chunk: "Chunk",
          parse: "Parse",
          vector: "Vector",
          error: "Error",
        },
      },
      search: {
        title: "Search Test",
        resultsTitle: "Search Results ({count})",
        score: "Score",
        metadata: "Metadata",
        noResults: "No relevant results found",
        typeLabel: "Search Type",
        topKLabel: "Top K",
        embeddingModelIdLabel: "Embedding Model",
        rerankModelIdLabel: "Rerank Model",
        rerankPlaceholder: "Optional",
        queryPlaceholder: "Enter search query...",
        searchButton: "Search",
        types: {
          hybrid: "Hybrid Search",
          dense: "Dense Search",
          sparse: "Sparse Search"
        }
      },
      edit: {
        title: "Edit Knowledge Base",
        description: "Modify knowledge base name",
        nameLabel: "Knowledge Base Name",
        namePlaceholder: "Enter new knowledge base name",
        updating: "Updating...",
        errors: {
          renameFailed: "Failed to rename knowledge base",
          updateFailed: "Failed to update knowledge base",
        },
      },
      errors: {
        deleteFailed: "Failed to delete document",
        pleaseSelectFiles: "Please select files first",
        uploadFailedWithName: "Failed to upload file: {name}",
        uploadFailedGeneric: "Upload failed",
        enterStartUrl: "Please enter a start URL",
        webImportFailed: "Website import failed",
        searchFailed: "Search failed",
        partialSuccess: "Partial success: {message}",
        deleteFailedWithMessage: "Delete failed: {message}",
      },
    },
  },
  build: {
    header: {
      backTitle: "Back to Builds",
      title: "Build",
      subTitle: "Build your application",
      status: {
        building: "Building",
        completed: "Completed",
        failed: "Failed"
      },
      deploy: {
        button: "Deploy",
        alertCompleteFirst: "Please complete a build first before deploying"
      }
    },
    page: {
      headerTitle: "Build Stage",
      headerDescription: "Convert Vibe interactive workflows into deployable applications",
      statusTitle: "Build Status",
      statusBadgeDev: "In Development",
      statusDevNote: "The Build stage is under development. It will support auto-generating deployable code from Vibe workflows in the future.",
      settingsButton: "Build Settings",
      stepsTitle: "Build Steps",
      steps: {
        codeGeneration: {
          title: "Code Generation",
          description: "Generate executable code from Vibe stage"
        },
        configBuild: {
          title: "Configure Build",
          description: "Set application configuration and dependencies"
        },
        packageDeploy: {
          title: "Package & Deploy",
          description: "Package the app and prepare for deployment"
        },
        apiGeneration: {
          title: "API Generation",
          description: "Generate RESTful API endpoints"
        },
        status: {
          pending: "Pending"
        }
      },
      featuresTitle: "Planned Features",
      features: {
        autoCode: {
          title: "Automated Code Generation",
          description: "Generate executable code from DAG workflows"
        },
        multiPlatform: {
          title: "Multi-platform Support",
          description: "Support Web, API, Mini Programs and more"
        },
        smartConfig: {
          title: "Smart Configuration",
          description: "Automatically generate config files and dependencies"
        }
      }
    },
    left: {
      title: "Input Configuration",
      appTypeLabel: "Application Type",
      labels: {
        relatedTask: "Related Task",
        selectTask: "Select Task to Build",
        loading: "Loading...",
        noTasksPrefix: "No tasks yet, please",
        createTask: "Create Task",
        goComplete: "Go to complete"
      },
      hints: {
        cannotBuildIncomplete: "Cannot build an incomplete task. Please complete it in VIBE first."
      }
    },
    actions: {
      cancel: "Cancel Build",
      viewHistory: "View History",
      start: "Start Build"
    },
    inputs: {
      title: "App Input Configuration",
      addText: "Text",
      addFile: "File",
      emptyHint: "Click the buttons above to add input fields",
      descriptionPlaceholder: "Describe this parameter",
      required: "Required"
    },
    model: {
      title: "Build Model",
      loading: "Loading model list...",
      empty: "No available LLM models, please add models in model management first",
      defaultSuffix: " (Default)"
    },
    process: {
      title: "Build Process",
      tdd: {
        title: "TDD Iteration Progress",
        attempt: "Attempt {attempt}/3"
      },
      logs: {
        title: "Build Logs",
        emptyHint: "Logs will appear after starting the build",
        building: "Building..."
      }
    },
    yaml: {
      title: "Generated YAML"
    },
    messages: {
      loadingHistory: "Loading build history records",
      loadHistoryFailedPrefix: "Failed to load history: {error}",
      startingBuild: "Starting build...",
      buildStarted: "Build started (Build #{number})",
      agentStarted: "Agent started executing...",
      execStepPrefix: "Execution step: ",
      analyzing: "Analyzing",
      invokeToolPrefix: "Invoking tool: ",
      toolResultReceived: "Tool result received",
      completed: "Build complete! YAML workflow generated",
      failedPrefix: "Build failed: {error}",
      cancelled: "Build cancelled",
      cancelling: "Cancelling build...",
      inputs: {
        configureFirst: "Please configure input fields on the left first",
      },
      tdd: {
        success: "TDD attempt {attempts} succeeded! YAML generated",
        failed: "TDD attempt {attempts} failed: {error}",
      },
      llm: {
        calling: "LLM calling...",
        responseReceived: "LLM response received",
      },
      ws: {
        noToken: "No authentication token found for WebSocket",
        connected: "Connected to build server",
        error: "WebSocket connection error",
        closed: "WebSocket connection closed",
      },
    },
    test: {
      title: "Test Area",
      ready: "Ready",
      notBuilt: "Not Built",
      hintReady: "Test the generated YAML workflow",
      hintNotReady: "Please build the app to enable testing",
      waiting: "Waiting for build to complete",
      waitingHint: "You can test app functionality here once the build completes",
      testing: "Testing...",
      testingHint: "Executing app using generated YAML",
      result: {
        success: "Test Success",
        failed: "Test Failed"
      },
      executionLog: "Execution Logs",
      finalResult: "Final Result",
      errorInfo: "Error Information",
      inputHint: "Enter parameters below and click Test",
      input: {
        title: "Test Input",
        placeholderPrefix: "Input ",
        submit: "Test"
      },
      messages: {
        successLog: "Test successful! Executed {nodes} nodes in {seconds} seconds",
        failedLog: "Test failed: {error}",
        execError: "Test execution error: {message}"
      },
      deployTitle: "Deploy This Application",
      deployHint: "Deploy button requires: success={success}, buildId={id}"
    },
    appTypes: {
      website: "Website",
      miniprogram: "WeChat Mini Program",
      api: "API"
    }
  },
  builds: {
    header: {
      title: "Build History",
      description: "View and manage build history",
      newBuild: "New Build"
    },
    filters: {
      search_placeholder: "Search by Build ID or Task name...",
      status: {
        all: "All Statuses"
      }
    },
    stats: {
      total: "Total {count} builds"
    },
    status: {
      started: "Started",
      building: "Building",
      completed: "Completed",
      failed: "Failed",
      cancelled: "Cancelled"
    },
    empty: {
      none: "No build records yet",
      createFirst: "Create first Build",
      noMatch: "No matching build records found"
    },
    card: {
      appType: "Application Type",
      createdAt: "Created At",
      completedAt: "Completed At",
      taskFallback: "Task #{id}",
      taskIdLabel: "Task ID",
      viewDetail: "View Details",
      delete: "Delete",
      deleteConfirm: "Are you sure you want to delete this build?"
    },
    list: {
      header: {
        title: "My Agents",
        description: "Manage your custom agents",
        create: "Create Agent",
      },
      search: {
        placeholder: "Search agents...",
      },
      empty: {
        title: "No agents found",
        description: "Get started by creating your first agent",
        create: "Create Agent",
      },
      pagination: {
        summary: "Showing {from}-{to} of {total} agents",
        prev: "Previous",
        next: "Next",
      },
      status: {
        draft: "Draft",
        published: "Published",
      },
      actions: {
        chat: "Chat",
        unpublish: "Unpublish",
        edit: "Edit",
        deleteConfirm: "Are you sure you want to delete this agent?",
      },
      chat: {
        welcome: "Hello! I'm {name}. {description}\n\nYou can start a conversation directly, or select a suggested prompt below.",
        inputPlaceholder: "Type your message...",
        loading: "Loading agent...",
        notFound: "Agent not found",
        failed: "Failed to load agent",
        taskCreated: "Task created. Implementing WebSocket connection...",
        error: "Error: {message}",
        sendFailed: "Failed to send message. Please try again.",
      }
    },
    pagination: {
      summary: "Showing {from}-{to} of {total} builds",
      prev: "Previous",
      next: "Next"
    },
    editor: {
      header: {
        title: "Custom Agent",
        subtitle: "Configure your AI agent's behavior and capabilities.",
        create: "Create",
        update: "Update",
        publish: "Publish",
        unpublish: "Unpublish",
        creating: "Creating...",
        updating: "Updating...",
      },
      validation: {
        nameRequired: "Agent name is required",
        modelRequired: "Please select a general model",
      },
      success: {
        created: "Agent created successfully",
        createdDesc: "Your '{name}' has been created successfully",
      },
      error: {
        failed: "Failed to create agent",
        unknown: "An unknown error occurred",
      }
    },
    configForm: {
      logo: {
        label: "Logo",
        upload: "Upload Logo",
        remove: "Remove Logo",
      },
      name: {
        label: "Name",
        placeholder: "e.g. Coding Assistant",
      },
      description: {
        label: "Description",
        placeholder: "Describe what this agent does...",
      },
      instructions: {
        label: "Instructions",
        placeholder: "System instructions for the agent...",
      },
      executionMode: {
        label: "Execution Mode",
        react: {
          title: "Standard Mode (Recommended)",
          description: "Quick response, flexible handling. Suitable for most daily tasks.",
        },
        graph: {
          title: "Planning Mode",
          description: "Systematic breakdown, step-by-step execution. Ideal for complex multi-step tasks.",
        },
      },
      model: {
        label: "Model",
        noData: "No models available",
        types: {
          general: "General Model",
          smallFast: "Small & Fast",
          visual: "Visual Model",
          compact: "Compact Model",
        },
        tips: {
          general: "General purpose LLM suitable for most tasks, balancing performance and cost.",
          smallFast: "Fast response and lower cost, suitable for simple tasks or low latency scenarios.",
          visual: "Capable of image recognition and processing, suitable for image-related tasks.",
          compact: "Lightweight model suitable for resource-constrained or high-immediacy scenarios.",
        },
      },
      knowledgeBase: {
        label: "Knowledge Base",
        placeholder: "Select Knowledge Base",
        create: "Create New Knowledge Base",
        noData: "No Knowledge Base available",
        selected: "{count} selected",
      },
      skills: {
        label: "Skills",
        placeholder: "Select Skills",
        create: "Create New Skill",
        noData: "No Skills available",
      },
      tools: {
        label: "Tools",
        placeholder: "Select Tool Categories",
        noData: "No tools available",
        selectedCount: "{count} categories selected ({tools} tools)",
        categories: {
          basic: "Basic Tools",
          file: "File Operations",
          vision: "Vision",
          image: "Image Generation",
          knowledge: "Knowledge Base",
          mcp: "MCP Tools",
          browser: "Browser Automation",
          ppt: "PowerPoint",
          office: "Office Tools",
          specialImage: "Image Tools",
        },
      },
      suggestedPrompts: {
        label: "Suggested Prompts",
        description: "Provide example prompts that users can click to use directly",
        placeholder: "Example prompt {index}",
        add: "+ Add Prompt",
        delete: "Delete",
      },
      actions: {
        save: "Save Configuration",
        reset: "Reset",
      }
    },
    preview: {
      title: "Preview",
      initialMessage: "Hello! I am the preview of your agent. You can test your configuration here.",
      chat: {
        placeholder: "Chat with your agent...",
        send: "Send",
        you: "You",
        agent: "Test Agent",
      },
      errors: {
        noModel: "Please select a general model first",
        requestFailed: "Preview request failed, please check network connection",
      }
    }
  },
  agent: {
    header: {
      backTitle: "Back to Vibe",
      badge: {
        process: "Process",
        task: "Task",
      },
      connection: {
        connected: "Connected",
        disconnected: "Disconnected",
        errorPrefix: "Connection error: {message}",
      },
      buildButton: "Build App",
    },
    build: {
      onlyProcessMode: "Only process mode can build apps",
      onlyProcessModeShort: "Process mode only",
    },
    initial: {
      title: "Start your task",
      subtitle: "Choose a mode and describe what you want to accomplish. AI will plan and execute for you.",
    },
    input: {
      placeholder: {
        default: "Describe your task requirements in detail...",
        continueExecution: "Enter adjustment instructions to modify the execution plan...",
      },
      actions: {
        config: "Configure",
        uploadFile: "Upload File",
        sendWithFiles: "Send and upload files",
        sendMessage: "Send Message",
        pauseTask: "Pause Task",
        resumeTask: "Resume Task",
      },
      status: {
        labels: {
          pending: "Pending",
          running: "Running",
          completed: "Completed",
          failed: "Failed",
          paused: "Paused",
        },
        selectedFiles: "Selected {count} files",
        taskStatusPrefix: "Task Status: ",
        processing: "Processing...",
      },
    },
    config: {
      title: "Configure Model",
    },
    configDialog: {
      title: "Agent Configuration",
      description: "Configure the model parameters for the Agent. Changes take effect immediately (current session only).",
      modelSelect: {
        label: "Select Model",
        description: "Choose an AI model from the added models to process your tasks",
        loading: "Loading model list...",
        empty: {
          title: "No available models",
          hint: "Please add models on the Models management page first",
        },
        main: {
          label: "Primary Model",
          placeholder: "Select primary model...",
          hint: "Primarily used for complex tasks. Defaults to the default model set on the Models page",
        },
        smallFast: {
          label: "Small/Fast Model (Optional)",
          placeholder: "Select small/fast model...",
          hint: "Optional configuration for simple tasks to get faster responses",
          options: {
            noneLabel: "Do not use fast model",
            noneDescription: "All tasks will be processed by the primary model",
            tagFast: " • Fast",
          },
        },
        visual: {
          label: "Vision Model (Optional)",
          placeholder: "Select vision model...",
          hint: "Optional configuration for image understanding and processing",
          options: {
            noneLabel: "Do not use vision model",
            noneDescription: "All tasks will be processed by the primary model",
            tagVisual: " • Vision",
          },
        },
        compact: {
          label: "Long Context Model (Optional)",
          placeholder: "Select long context model...",
          hint: "Optional configuration used for long context compression and processing",
          options: {
            noneLabel: "Do not use long context model",
            noneDescription: "All tasks will be processed by the primary model",
            tagCompact: " • Long Context",
          },
        },
      },
      memoryThreshold: {
        label: "Memory similarity threshold",
        subLabel: "Threshold",
        strict: "Strict",
        loose: "Loose",
        hint: "Adjust the threshold for recalling related memories. Lower values mean stricter matching; higher values recall more memories.",
      },
      buttons: {
        close: "Close",
      },
    },
    process: {
      execute: "Start Process",
    },
    hints: {
      taskMode: "Press Enter to send • Shift+Enter for newline • Markdown supported",
      processMode: "Fill in the process description, then click Execute",
    },
    status: {
      running: "Running",
      completed: "Completed",
      failed: "Failed",
      pending: "Pending",
      paused: "Paused",
    },
    planDetails: {
      collapsibleTitle: "Details",
      badge: {
        memory: "Memory",
        plan: "Plan",
      },
      memory: {
        title: "Memory Info",
        stats: {
          found: "Found: {count}",
          used: "Used: {count}",
        },
        enhancedGoalTitle: "Enhanced Goal:",
        relatedTitle: "Related Memories:",
        unknownCategory: "Unknown Category",
      },
      plan: {
        title: "Execution Plan",
        goalTitle: "Goal:",
        stepsTitle: "Steps ({count}):",
        dependenciesPrefix: "Dependencies: ",
      },
    },
    vibeMode: {
      tabs: {
        task: "Task Mode",
        process: "Process Mode",
      },
      descriptions: {
        task: {
          title: "Task Mode",
          text: "One-off task mode to quickly achieve specific goals. Suitable for exploration and experimentation.",
          examplesTitle: "Examples:",
          examplesText: "Complete PPT writing, generate posters, build websites, etc.",
        },
        process: {
          title: "Process Mode",
          text: "Reusable agent workflows suitable for Build and Deploy. Requires defining a process description and input/output examples.",
          examplesTitle: "Examples:",
          examplesText: "Describe customer service bot process, audit assistant, etc.",
          examples: {
            apiExample: {
              title: "Token Refresh Demo",
              labels: {
                currentUser: "Current user:",
                accessTokenStatus: "Access Token status:",
                refreshTokenStatus: "Refresh Token status:",
                wsStatus: "WebSocket status:",
                wsError: "WebSocket error:",
              },
              status: {
                obtained: "Obtained",
                notObtained: "Not obtained",
              },
              autoApi: {
                title: "API call using auto token refresh",
              },
              common: {
                errorPrefix: "Error:",
              },
              models: {
                success: "Successfully fetched {count} models",
                refetch: "Refetch",
              },
              manualApi: {
                title: "Manual API call (for comparison)",
                button: "Call API manually",
                failed: "API call failed",
              },
              guide: {
                title: "Dual token mechanism explanation:",
                accessToken: "Access Token: expires in 30 minutes, used for API calls",
                refreshToken: "Refresh Token: expires in 7 days, used to refresh access token",
                autoRefresh: "Auto refresh: refreshes 1 minute before access token expiry",
                offlineRecovery: "Offline recovery: come back after hours and still auto restore login",
                wsSupport: "WebSocket support: WebSocket connections also support auto token refresh",
                tokenRotation: "Token rotation: each refresh generates a new refresh token to improve security",
                seamless: "Seamless experience: users never need to re-login due to token expiry",
                smartCache: "Smart cache: prioritize checking refresh token expiry time",
              },
            },
          },
        },
      },
      form: {
        processDescription: {
          label: "Process Description",
          placeholder: "Describe in detail the purpose, applicable scenarios, and expected behavior of this agent process...",
        },
        examples: {
          label: "Input/Output Examples",
          optional: "(Optional)",
          add: "Add Example",
          badge: "Example {index}",
          input: {
            label: "Input:",
            placeholder: "Enter example input...",
          },
          output: {
            label: "Output:",
            placeholder: "Enter example output...",
          },
        },
      },
    },
    layout: {
      common: {
        notStarted: "Not started",
        inProgress: "In progress",
        unknown: "Unknown",
      },
      status: {
        pending: "Pending",
        running: "Running",
        completed: "Completed",
        failed: "Failed",
        skipped: "Skipped",
      },
      right: {
        titles: {
          stepDetail: "Step Details",
          executionLogs: "Execution Logs",
        },
        labels: {
          resultData: "Result Data",
          stepData: "Step Data",
          fileOutputs: "File Outputs",
          executionTime: "Execution Time",
          currentStep: "Current Step:",
          none: "None",
        },
        buttons: {
          previewAll: "Preview All",
          prevStep: "Previous",
          nextStep: "Next",
        },
        tooltips: {
          previewAllFiles: "Preview all files",
          previewFile: "Preview file",
        },
        branch: {
          conditionalNode: "Conditional Branch Node",
          optionalBranches: "Optional Branches",
          branchCondition: "Branch Condition",
          requiredBranch: "Required Branch",
        },
        counts: {
          files: "{count} files",
          logs: "{count} logs",
          fileOutputsWithCount: "File Outputs ({count})",
        },
        empty: {
          selectStepHint: "Select a step to view details",
          noLogs: "No execution logs yet",
        },
        misc: {
          unknownFile: "Unknown File",
        },
      },
      center: {
        titles: {
          dag: "DAG Execution Graph",
        },
        labels: {
          vertical: "Vertical",
          horizontal: "Horizontal",
          layoutVerticalTitle: "Vertical layout (Top-Bottom)",
          layoutHorizontalTitle: "Horizontal layout (Left-Right)",
          phase: "Execution Phase",
          updatedAt: "Updated:",
          status: "Status:",
          conditionalBranchNode: "Conditional Branch Node",
          optionalBranches: "Optional Branches",
          branchCondition: "Branch Condition",
          requiredBranch: "Required Branch:",
          branches: "Branches:",
          tools: "Tools:",
          pureAnalysis: "Pure Analysis",
          description: "Description:",
          result: "Execution Result",
        },
        planning: {
          title: "AI is creating an execution plan",
          subtitle: "Analyzing the task and constructing the optimal execution path",
          steps: {
            analyze: "Analyze requirements",
            buildGraph: "Build execution graph",
            optimizePath: "Optimize path",
          },
          spinner: "Building execution graph...",
        },
        time: {
          title: "Time Information",
          startedAt: "Start Time:",
          completedAt: "Completed Time:",
          duration: "Duration:",
          inProgress: "In progress",
        },
        errors: {
          taskFailedTitle: "Task Execution Failed",
          taskFailedDesc: "An error occurred during execution. Please check the messages on the left for details.",
          checkLeft: "Check error details in the left messages",
          retryTask: "Try resubmitting the task",
        },
      },
      left: {
        title: "Task Execution",
        messageTags: {
          result: "Result",
          file: "File",
        },
      },
    },
    title: "Agent",
    taskId: "Task ID: {id}",
    logs: {
      event: {
        labels: {
          start: "Start",
          success: "Success",
          completed: "Completed",
          failed: "Failed",
          llmCall: "LLM Call",
          llmCompleted: "LLM Completed",
          llmFailed: "LLM Failed",
          toolCall: "Tool Call",
          toolCompleted: "Tool Completed",
          toolFailed: "Tool Failed",
          memoryGenerate: "Memory Generate",
          memoryStore: "Memory Store",
          memoryQuery: "Memory Query",
          compactStart: "Compact Start",
          compactCompleted: "Compact Completed",
          skillSelectStart: "Skill Select",
          skillSelectEnd: "Skill Selected",
          unknown: "Unknown Operation",
        },
        actions: {
          stepStart: "Step Start",
          stepCompleted: "Step Completed",
          stepFailed: "Step Failed",
          llmStart: "LLM Call Start",
          llmCompleted: "LLM Call Completed",
          llmInfo: "LLM Call Info",
          llmResult: "LLM Call Result",
          llmFailed: "LLM Call Failed",
          toolStart: "Tool Call Start",
          toolCompleted: "Tool Call Completed",
          toolFailed: "Tool Call Failed",
          useTool: "Use Tool",
          compactStepErrorLog: "Step Error Log",
          reactTaskStart: "ReAct Task Start",
          reactTaskCompleted: "ReAct Task Completed",
          reactStepStart: "ReAct Step Start",
          reactStepCompleted: "ReAct Step Completed",
          memoryGenerateStart: "Memory Generate Start",
          memoryGenerateCompleted: "Memory Generate Completed",
          memoryStoreStart: "Memory Store Start",
          memoryStoreCompleted: "Memory Store Completed",
          memoryQuery: "Memory Query",
          memoryQueryCompleted: "Memory Query Completed",
          memoryQueryStart: "Memory Query Start",
          compactStart: "Context Compact Start",
          compactCompleted: "Context Compact Completed",
          skillSelectStart: "Skill Select Start",
          skillSelectEnd: "Skill Select End",
          skillName: "Skill Name",
        },
        messages: {
          planStart: "Start generating execution plan: {phase}",
          planEnd: "Execution plan completed (Plan ID: {planId}, Steps: {stepsCount})",
          taskStart: "Start executing task (Iteration {iteration})",
          taskEnd: "Task execution completed (Iteration {iteration})",
          taskDesc: "Task: {taskPreview}",
          compactStart: "🗜️ Context compact start",
          compactCompleted: "🗜️ Context compact completed",
          stepFailed: "Step failed: {stepName}",
          stepSkipped: "⏭️ Step skipped: {stepName}",
          finalAnswerGenerating: "Generating final answer...",
          finalAnswerCompleted: "Final answer generated",
          goalCheckStart: "Start goal check...",
          goalCheck: "Goal Check",
          goalAchieved: "Achieved",
          goalNotAchieved: "Not achieved",
          confidence: "(Confidence: {percent}%)",
          reason: "Reason:",
          reasonLabel: "Reason: {goalReason}",
          memoryWillStore: "Will store execution memory: {memoryReason}",
          taskLLMStart: "Start task-level LLM call: {taskType}",
          taskLLMCompleted: "Task-level LLM call completed: {taskType}",
          planLLMSending: "💭 Plan generation: sending request to LLM ({modelName})",
          planLLMResponseCompleted: "Plan generation: LLM response completed ({modelName})",
          useTool: "🔧 Using tool: {toolName}",
          metaTitle: "Meta Info:",
          fileOutputsGenerated: "Generated {count} output files:",
          previewLabel: "Preview",
          executionResultPrefix: "Execution Result:",
          errorPrefix: "Execution Error:",
          visualUpdate: "Visualization Update: {type}",
          reactTaskStart: "🔄 ReAct task start",
          reactTaskCompleted: "🔄 ReAct task completed",
          reactStepStart: "🔄 ReAct step start: {stepName}",
          reactStepCompleted: "🔄 ReAct step completed: {stepName}",
          detailsTitle: "Details",
          memoryBadge: "Memory",
          storeSuggestion: "Store Suggestion:",
          worthStoring: "Worth storing",
          notWorthStoring: "Not worth storing",
          insightsLabel: "Insights generated:",
          storageStatusLabel: "Storage status:",
          notStored: "Not stored",
          decisionLabel: "Decision:",
          taskLabel: "Task:",
          taskInfoLabel: "Task Info:",
          categoryUnknown: "Unknown category",
          stepsCount: "Steps: {count}",
          goalTitle: "Goal:",
          execStepPrefix: "step",
        },
        common: {
          fullData: "Full Data",
          itemsSuffix: " items",
        },
        llm: {
          model: "Model",
          contextMessages: "Context Messages",
          paramsTitle: "Call Parameters",
          temperature: "Temperature:",
          maxTokens: "Max Tokens:",
          topP: "Top P:",
        },
        tool: {
          tool: "Tool",
          paramsCount: "Params Count",
          paramsTitle: "Tool Params",
        },
        compact: {
          type: "Compact Type",
          model: "Compact Model",
          originalTokens: "Original Tokens",
          threshold: "Threshold",
          resultTitle: "Compact Result",
          compactedTokens: "Compacted Tokens",
          compressionRatio: "Compression Ratio",
          errorTitle: "Compact Error",
          types: {
            individual_dependency: "Individual Dependency",
            entire_context: "Entire Context",
          },
        },
        memory: {
          task: "Query Task",
          category: "Memory Category",
          found: "Memory Found",
          used: "Memory Used",
          relatedTitle: "Related Memories",
        },
        skill: {
          task: "Task",
          availableCount: "Available Skills",
          selected: "Selected",
          yes: "Yes",
          no: "No",
          skillName: "Skill Name",
          skill: "available skill",
          skills: "available skills",
        },
      },
      messagesPreview: {
        title: "Context Messages",
        infoTitle: "Context Info",
        actions: {
          expand: "Expand",
          collapse: "Collapse",
        },
        labels: {
          role: {
            system: "System",
            user: "User",
            assistant: "Assistant",
          },
          messageIndex: "Message #{index}",
        },
        parse: {
          error: "Parse Error",
          failed: "Parse Failed",
          errorPrefix: "Parse Error:",
        },
      },
    },
    replay: {
      controls: {
        stop: "Stop Replay",
        pause: "Pause",
        play: "Play",
        speed: "Playback Speed",
        speedSuffix: "x",
      },
    },
  },
  deployments: {
    header: {
      title: "Deployments",
      description: "Manage and monitor deployed agents and their endpoints",
      create: "Create Deployment",
    },
    dialog: {
      createTitle: "Create Deployment",
      selectBuild: "Select Build",
      selectPlaceholder: "Select a build...",
      name: "Deployment Name",
      namePlaceholder: "Enter a deployment name...",
      descriptionOptional: "Description (Optional)",
      descriptionPlaceholder: "Enter description (optional)",
      creating: "Creating...",
      formRequired: "Please select a build and enter a deployment name",
    },
    status: {
      running: "Running",
      stopped: "Stopped",
      creating: "Creating",
      error: "Error",
    },
    empty: {
      title: "No deployments yet",
      description: "Create your first deployment to publish your agent",
    },
    card: {
      executions: "executions",
      last: "Last:",
      created: "Created:",
      copy: "Copy",
      stop: "Stop",
      start: "Start",
      details: "Details",
      deleteConfirm: "Are you sure you want to delete this deployment?",
      never: "Never",
    },
    messages: {
      fetchDeploymentsFailed: "Failed to load deployments",
      fetchBuildsFailed: "Failed to load builds",
      createFailed: "Failed to create deployment",
      startFailed: "Failed to start deployment",
      stopFailed: "Failed to stop deployment",
      deleteFailed: "Failed to delete deployment",
      endpointCopied: "Endpoint copied",
    },
    detail: {
      backTitle: "Back to Deployments",
      notFound: "Deployment not found",
      goBack: "Go back to Deployments",
      info: {
        title: "Deployment Info",
        status: "Status:",
        type: "Type:",
        executions: "Executions:",
        lastExecuted: "Last executed:",
        created: "Created:",
        build: "Build:",
        viewBuild: "View Build",
        endpointLabel: "Endpoint URL:",
      },
      test: {
        title: "Test Execution",
        placeholder: "Enter your test query or task...",
        inputPlaceholder: "Enter {name}...",
        button: {
          execute: "Execute",
          executing: "Executing...",
        },
        resultTitle: "Result:",
        apiUsage: {
          title: "API Usage:",
          endpoint: "Endpoint:",
          requestBody: "Request Body:",
          example: "Example:",
          yourTaskHere: "your task here",
          yourNameHere: "your {name} here",
        },
      },
      yaml: {
        title: "YAML Configuration",
        empty: "No YAML content available",
      },
      executions: {
        title: "Execution History",
        empty: "No executions yet",
        task: "Task:",
        threadId: "Thread ID:",
        output: "Output:",
        timeNodes: "Time: {seconds}s | Nodes: {nodes}",
        error: "Error:",
      },
      misc: {
        na: "N/A",
        deleteExecutionConfirm: "Are you sure you want to delete this execution record?",
        errorPrefix: "Error:",
      },
      messages: {
        fetchFailed: "Failed to fetch deployment detail",
        fetchExecutionsFailed: "Failed to fetch execution history",
        executionFailed: "Execution failed",
        inputRequired: "Please enter a value for {name}",
      },
    },
  },
  traceEventRenderer: {
    skillSelected: "Selected Skill",
    taskExecution: "Task Execution",
    unknownModel: "Unknown Model",
    callLLM: "Call LLM: {model}",
    llmResponse: "LLM Response",
    unknownTool: "Unknown Tool",
    executeTool: "Execute Tool: {tool}",
    toolExecutionFinished: "Tool Execution Finished",
    unknownError: "Unknown error",
    executionFailed: "Execution Failed",
    view: "View",
    codeAndOutput: "Code & Output",
    args: "Args",
    output: "Output",
    executing: "Executing...",
    errorLabel: "Error: ",
    executionCode: "Execution Code:",
    outputResult: "Output Result:",
    toolLabel: "Tool: ",
    argumentsLabel: "Arguments:",
    codeLabel: "Code:",
    outputLabel: "Output:",
    modelLabel: "Model: ",
    reasoningLabel: "Reasoning:",
    noReasoning: "(No reasoning yet)",
    toolCallsLabel: "Tool Calls:",
    errorTitle: "Error:"
  },
} as const

export default en
