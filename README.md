<div align="center">

![Xagent Banner](./assets/github_readme_banner.jpg)

[![Discord](https://img.shields.io/discord/1474756736358289609?style=for-the-badge&logo=discord)](https://discord.gg/R7TDFMzuXq)
[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/+2_-SAVLtuJNkNWFl)
[![Twitter](https://img.shields.io/twitter/follow/xorbitsio?logo=x&style=for-the-badge)](https://twitter.com/xorbitsio)

[![Documentation](https://img.shields.io/badge/docs-docs.xagent.run-blue?style=for-the-badge&logo=gitbook)](https://docs.xagent.run/)
[![GitHub Release](https://img.shields.io/github/v/release/xorbitsai/xagent?logo=github&style=for-the-badge)](https://github.com/xorbitsai/xagent/releases)
[![Docker Pulls](https://img.shields.io/docker/pulls/xprobe/xagent-backend?style=for-the-badge&logo=docker)](https://hub.docker.com/r/xprobe/xagent-backend)

</div>

---

## What is Xagent?

**Describe tasks. Not workflows.**

**No more flowcharts.**
**No more rigid automation.**
**Just tell Xagent what you want.**

👏 Join [Telegram](https://t.me/+2_-SAVLtuJNkNWFl) | [Discord](https://discord.gg/R7TDFMzuXq)

❤️ Like Xagent? Give it a star 🌟 to support the development!

---

## Xagent vs OpenClaw

| Capability         | Xagent                                                        | OpenClaw                                                                                  |
| ------------------ | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Core design        | **Enterprise agent platform**                                 | Autonomous personal agent                                                                 |
| Security           | **VM‑level sandbox** for safe agent execution                 | Only a Docker container sandbox — and real usage often requires elevated host permissions |
| Agent architecture | **LLM‑driven planning + efficient multi‑agent orchestration** | Mostly linear task execution                                                              |
| Model capability   | **API-compatible* + model-native intelligence**               | API or local models used as simple plug-ins (no deep integration)                        |
| Knowledge systems  | **Enterprise RAG / knowledge platforms**                      | Local memory / lightweight RAG                                                            |
| Deployment         | **Flexible deployment** — local, private (on‑prem), or cloud  | Mostly local runtime                                                                      |
| Multi-tenancy      | **Tenant‑aware architecture**                                 | Primarily single‑user                                                                     |

\* Xagent supports both API-based models and open-source models, powered by deep integration with Xinference for model-level optimization.

---

## 🎬 See Xagent Think

Give it a goal.
Watch it plan, select tools, execute, and deliver.

![Xagent Demo](./assets/task_demo.jpg)

---

## ⚡ The Problem

**Workflow builders are rigid. They break when requirements change.**

- You map every decision branch manually
- You orchestrate tools by hand
- You maintain fragile flow diagrams
- You re-engineer when logic changes

---

## 🎯 The Xagent Way

**With Xagent, you describe the outcome — not the steps.**

- Plans the task dynamically
- Decomposes into executable steps
- Selects the right tools automatically
- Executes, evaluates, and iterates

---

## 🚀 What You Can Build

With Xagent, you don’t design workflows. You describe the task. That’s it.

**Just define what you want done — and Xagent plans, decomposes, selects tools, and executes.**

### Build Anything You Can Describe:

| Domain | Examples |
|--------|----------|
| **Content Creation** | Automated PPT generation, marketing posters, creative design assets |
| **Research & Analysis** | Deep analysis reports, research summaries, data synthesis |
| **Enterprise Automation** | AI copilots for internal teams, knowledge assistants, task automation |
| **Business Intelligence** | Data reporting automation, SaaS features, multi-step reasoning systems |
| **Knowledge Work** | Document processing, information extraction, structured outputs |

**If you can clearly describe the goal, Xagent can turn it into an executable system.**

All powered by one unified runtime.

---

## 🚀 Quick Start

**Get started in 3 minutes**

### 1️⃣ Clone and configure

```bash
git clone https://github.com/xorbitsai/xagent.git
cd xagent
cp example.env .env
```

### 2️⃣ Start with Docker

```bash
docker compose up -d
```

To enable sandbox (requires Linux or WSL2 with KVM support):

```bash
docker compose -f docker-compose-sandbox.yml up -d
```

### 3️⃣ Open in browser

```
http://localhost:80
```

On first startup, Xagent redirects to `/setup`.

Create the first administrator account there to complete initialization.

If the admin password is forgotten, reset it via CLI:

```bash
python -m xagent.web.reset_admin_password --username <admin_username>
```

That's it. Xagent is now running.

---

## ✨ Core Features

### 🧠 Dynamic Planning Engine

Unlike traditional workflow tools, Xagent plans tasks dynamically at runtime.

- Automatic task decomposition
- Plan → Execute → Reflect loops
- Conditional branching
- Multi-step reasoning

**No static flows. No brittle chains.**

---

### 🔌 Tool & Model Orchestration

Xagent connects to your entire stack:

- OpenAI, Anthropic and other LLM providers
- Self-hosted models via Xinference
- External APIs
- Knowledge bases (RAG)
- Internal enterprise systems

It selects and orchestrates tools automatically during execution.

---

### ⚡ Instant Execution Mode

For simple use cases, run tool-enabled LLM calls instantly.

- No configuration overhead
- Chat-style assistants
- Embedded AI features

**Start simple. Scale when needed.**

---

### 📊 Observability & Control

Built for real production use:

- Task lifecycle tracking
- Token usage monitoring
- Execution state management
- Multi-user support

**Operate agents like real systems — not demos.**

---

## 🎬 Xagent in Action

Watch Xagent plan, execute, and deliver results in real-time.

![Xagent in Action](./assets/task.gif)

---

## Stay Ahead

Xagent is actively developed and rapidly evolving.

![Stay Ahead](./assets/xagent_stay_ahead.gif)

**Follow our progress:**
- ⭐ Star us on GitHub to stay updated
- 🐛 Report issues and request features
- 💬 Join our community discussions

---

## 🏢 Deployment Options

Xagent supports:

- Self-hosted deployment
- Private cloud environments
- On-premise enterprise infrastructure
- Docker-based setup

**You control your models, data, and infrastructure.**

---

## 🏗 Architecture Overview

Xagent separates core responsibilities:

| Layer | Responsibility |
|-------|----------------|
| **Agent Definition** | Intent & constraints |
| **Planning Engine** | Dynamic decomposition |
| **Execution Runtime** | Orchestration layer |
| **Tool Layer** | Integrations & actions |
| **Model Layer** | LLM & inference backend |

**This architecture enables:**
- Stability under complex reasoning
- Safe iteration
- Horizontal scalability
- Long-term maintainability

---

## 🔍 Not a Workflow Builder

**Xagent is not:**
- A drag-and-drop flow editor
- A static template engine
- A chatbot wrapper

**Xagent is:**
- A dynamic task execution engine
- An autonomous planning system
- A foundation for building real AI agents

---

## 🤝 Contributing

We welcome developers, product builders, and researchers.

Open issues. Submit PRs. Help shape the future of AI agents.

---

## 💬 Community & Contact

**[Documentation](https://docs.xagent.run/)** - Full documentation and guides

**[GitHub Issues](https://github.com/xorbitsai/xagent/issues)** - Report bugs or propose features

**[Discord](https://discord.gg/R7TDFMzuXq)** - Share your tasks or agents and connect with the community

**[Telegram](https://t.me/+2_-SAVLtuJNkNWFl)** - Join our Telegram group for discussions

**[X (Twitter)](https://twitter.com/xorbitsio)** - Follow for updates and share your work

---

## 📄 License

This project is licensed under the Xagent Source License - see the [LICENSE](LICENSE) file for details.
