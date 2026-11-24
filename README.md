# WW-P Security AI Agent

### Local AI-Powered Cybersecurity & System Administration Assistant

**WW-P Security AI Agent** is an evolving, security-oriented AI agent designed to support cybersecurity analysis, system administration, and intelligent IT operations.

The project is being developed by extending a secure Retrieval-Augmented Generation (RAG) architecture with local Large Language Models (LLMs), controlled tool execution, and an agent-based reasoning architecture.

Its primary objective is to combine local AI inference, secure knowledge retrieval, role-based access control, and controlled automation into a modular AI operations platform.

> **Project status:** Active Development  
> **Development date:** November 24, 2026  
> **Current focus:** Secure RAG, Ollama integration, Tool Registry, and Agent Tool Bridge.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Vision and Objectives](#vision-and-objectives)
- [Architecture](#architecture)
- [Implemented Components](#implemented-components)
- [Current Development Progress](#current-development-progress)
- [Technology Stack](#technology-stack)
- [Security Architecture](#security-architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
- [Testing](#testing)
- [Development Roadmap](#development-roadmap)
- [Planned Agent Capabilities](#planned-agent-capabilities)
- [Project Limitations](#project-limitations)
- [Author](#author)

---

## Project Overview

WW-P Security AI Agent aims to provide an intelligent assistant capable of:

- Analyzing cybersecurity-related information.
- Interpreting technical documentation.
- Retrieving authorized information from private knowledge bases.
- Communicating with locally hosted LLMs.
- Selecting and executing authorized tools.
- Supporting IT troubleshooting and system diagnostics.
- Assisting with network and security monitoring.

Unlike a conventional chatbot, the planned system is designed around controlled tool execution and a modular agent architecture.

Security is a fundamental design requirement rather than an additional feature.

The project follows a local-first approach, using Ollama to run language models locally.

---

## Vision and Objectives

The long-term objective is to build a modular AI agent combining five major capabilities:

1. **Secure Knowledge Retrieval** – retrieving information from authorized documents.
2. **Local AI Intelligence** – using locally hosted language models.
3. **Controlled Tool Execution** – interacting with system and security tools through permission-controlled interfaces.
4. **Agent Reasoning** – planning and executing multi-step tasks.
5. **Security and Accountability** – enforcing permissions, validating operations, and maintaining audit records.

The architecture is intended to support both cybersecurity and general IT operations.

---

## Architecture

The system is being developed as a collection of independent modules.

### Target Architecture

```text
             USER INTERFACE
                    |
                    v
             AGENT CORE
                    |
          +---------+---------+
          |                   |
          v                   v
      LLM CLIENT          PLANNER
          |                   |
          v                   v
       OLLAMA             TASK PLAN
          |                   |
          +---------+---------+
                    |
                    v
              TOOL REGISTRY
                    |
             PERMISSION CHECK
                    |
                    v
               TOOL BRIDGE
                    |
                    v
               TOOL EXECUTOR
                    |
          +---------+----------+
          |         |          |
          v         v          v
       Security   Log       System
       Analyzer  Analyzer  Diagnostics
                    |
                    v
              TOOL RESULTS
                    |
                    v
             AGENT OBSERVATION
                    |
                    v
             FINAL RESPONSE
```

The diagram represents the target architecture. Not all components and execution loops have been implemented yet.

### Secure RAG Architecture

```text
       DOCUMENT INPUT
              |
              v
       DOCUMENT LOADER
              |
              v
        TEXT EXTRACTOR
              |
              v
          CHUNKER
              |
              v
          EMBEDDER
              |
              v
       ENCRYPTED FAISS
              |
              v
       SECURE RETRIEVER
              |
              v
         RBAC / ACL
              |
              v
       AUTHORIZED CONTEXT
              |
              v
         LLM CLIENT
```

---

## Implemented Components

### 1. Authentication and Authorization

The security foundation includes:

- User authentication.
- Password hashing with Argon2.
- Role-Based Access Control (RBAC).
- Document-level Access Control Lists (ACL).
- User context management.

The purpose is to ensure that users can only retrieve information they are authorized to access.

### 2. Encrypted Vector Database

The project uses FAISS for vector similarity search.

Security mechanisms include:

- AES-256-GCM encryption.
- Encrypted vector index storage.
- Integrity verification.
- Encrypted document content and metadata.
- Secure index persistence.
- Separation of document identifiers and vector identifiers.
- Atomic persistence operations.
- Avoidance of unsafe pickle-based deserialization.

The embedding model currently used in the RAG foundation is:

`sentence-transformers/all-MiniLM-L6-v2`

Embedding dimension: 384.

### 3. Secure Retriever

Secure Retriever is responsible for retrieving relevant information while respecting authorization rules.

Its responsibilities include:

- Semantic similarity search.
- Document authorization.
- ACL enforcement.
- Filtering unauthorized information.
- Preparing authorized context for the LLM.

### 4. PromptGuard

PromptGuard provides an initial layer of protection against potentially malicious or unsafe prompts.

It analyzes incoming user requests using configurable security rules.

### 5. JailbreakDetector

JailbreakDetector is designed to identify attempts to bypass the intended security restrictions of the system.

It complements PromptGuard as part of the input security pipeline.

### 6. Ollama Integration

The project is integrating Ollama as its local LLM inference provider.

Installed models:

- `qwen2.5:3b`
- `phi3:latest`

The Ollama client has been extended to support:

- `/api/generate`
- `/api/chat`
- Chat messages.
- Tool definitions.
- Tool call parsing.
- Structured tool-call responses.

The Qwen 2.5 3B model has been verified to advertise tool-calling capabilities through Ollama.

The implementation of the complete agent-side tool execution cycle is still in progress.

### 7. Tool Registry and Execution Foundation

The project is developing a controlled tool execution architecture.

Its components include:

- Tool Registry.
- Tool Permissions.
- Tool Executor.
- Tool Bridge.

The objective is to prevent the LLM from executing arbitrary operations directly.

Instead, the model should request available tools through a controlled interface, where permissions and input validation can be enforced.

---

## Current Development Progress

**Development snapshot: November 24, 2026**

| Component | Status |
|---|---|
| Authentication | Implemented |
| Argon2 Password Hashing | Implemented |
| RBAC | Implemented |
| Document ACL | Implemented |
| Encrypted FAISS | Implemented |
| AES-256-GCM | Implemented |
| Secure Retriever | Implemented |
| PromptGuard | Implemented |
| JailbreakDetector | Implemented |
| Ollama Client | Implemented / Under Testing |
| Ollama Chat API | Implemented |
| Tool Call Parser | Under Testing |
| Tool Registry | Implemented / Under Testing |
| Tool Permissions | Implemented / Under Testing |
| Tool Executor | Implemented / Under Testing |
| Tool Bridge | In Development |
| Agent Core | Planned |
| Reason-Act-Observe Loop | Planned |
| Autonomous Planning | Planned |
| Agent Memory | Planned |
| Human-in-the-Loop | Planned |
| Audit Logging | Planned |

**Important:** Implemented modules do not necessarily mean that their complete end-to-end integration has been validated.

### Recent Testing Progress

The following results have been recorded during development:

| Test Area | Result |
|---|---|
| Existing regression suite | 72 passed |
| Ollama adapter tests | 40 passed |
| Parser tests | 41 passed, 1 failed |
| Extended regression | 154 passed (reported during development) |

The parser failure identified during development concerned the test helper's handling of an explicit `tool_calls=None` value.

Additional integration testing is required before considering the entire tool execution pipeline validated.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Local LLM Runtime | Ollama |
| Language Models | Qwen 2.5 3B, Phi-3 |
| Vector Database | FAISS |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Database | SQLite |
| Security | AES-256-GCM, Argon2 |
| Authorization | RBAC, ACL |
| AI Architecture | RAG, Tool Calling, Agent Architecture |
| Testing | Pytest |
| Version Control | Git, GitHub |

---

## Security Architecture

Security is one of the main architectural principles of WW-P Security AI Agent.

The project follows several important design principles:

**1. Least Privilege**

Users and tools should receive only the permissions necessary to perform their assigned tasks.

**2. Controlled Tool Execution**

LLM-generated tool calls must pass through a controlled execution layer.

**3. Access Control**

Document retrieval must respect user roles and document-level permissions.

**4. Data Protection**

Sensitive vector data and document information are protected using encryption.

**5. Input Validation**

Prompts and tool arguments should be checked before processing or execution.

**6. Human Oversight**

Sensitive operations are intended to require explicit human approval.

**7. Auditability**

Important operations should be recorded to support troubleshooting and security analysis.

These principles guide the ongoing development of the system. Some controls remain planned rather than fully implemented.

---

## Installation

### Requirements

- Python 3.10 or newer.
- Git.
- Ollama.
- Windows or Linux.
- Sufficient RAM for the selected local language model.

### 1. Clone the Repository

```bash
git clone https://github.com/PiotrIT2015/secure-rag.git

cd secure-rag
```

### 2. Create a Virtual Environment

Windows:

```powershell
python -m venv venv

.\venv\Scripts\Activate.ps1
```

Linux:

```bash
python3 -m venv venv

source venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### 4. Install Ollama

Download and install Ollama from:

https://ollama.com/

Download the selected models:

```bash
ollama pull qwen2.5:3b

ollama pull phi3:latest
```

Verify the installation:

```bash
ollama list
```

Check model capabilities:

```bash
ollama show qwen2.5:3b
```

---

## Configuration

The application requires configuration of its security credentials and local AI provider.

Configuration should be stored in environment variables or a local `.env` file.

Example configuration template:

```env
OLLAMA_HOST=http://localhost:11434

OLLAMA_MODEL=qwen2.5:3b

JWT_SECRET_KEY=your-generated-secret

DATABASE_URL=your-database-configuration
```

**Security recommendations:**

- Never commit `.env` files.
- Never publish authentication secrets.
- Generate unique cryptographic secrets.
- Use strong passwords.
- Protect encrypted database and vector-store files.
- Do not expose local AI services to untrusted networks.

The exact variable names should match the application's configuration implementation.

---

## Running the Project

The execution entry point depends on the currently integrated application version.

For the existing application:

```bash
python app.py
```

For test execution:

```bash
python -m pytest tests/ -v
```

---

## Testing

Testing is an important part of the development process.

The project uses Pytest to validate individual components and their interactions.

Example commands:

```bash
python -m pytest tests/test_tool_permissions.py -v
```

```bash
python -m pytest tests/test_ollama_tool_integration.py -v
```

```bash
python -m pytest tests/ -v
```

Testing priorities include:

- Authentication and authorization.
- Document access control.
- Vector database encryption.
- Retrieval correctness.
- Prompt injection detection.
- Jailbreak detection.
- Ollama communication.
- Tool-call parsing.
- Tool permission enforcement.
- Tool execution.
- Agent workflow security.

---

## Development Roadmap

The development process is divided into several stages.

| Stage | Description | Status |
|---|---|---|
| 1 | Secure RAG Foundation and Integration | In Progress |
| 2 | Ollama Integration | In Progress |
| 3 | Tool Registry and Permissions | In Progress |
| 4 | Agent Core | Planned |
| 5 | Reason → Act → Observe | Planned |
| 6 | Planning and Agent Memory | Planned |
| 7 | Human-in-the-Loop | Planned |
| 8 | Security Validation and Monitoring | Planned |
| 9 | Integration and End-to-End Testing | Planned |
| 10 | Documentation and Release | Planned |

### Immediate Development Priorities

1. Complete the Ollama tool-call integration tests.
2. Finalize Tool Bridge.
3. Verify the complete Tool Registry → Tool Executor execution path.
4. Connect the LLM to the tool execution layer.
5. Implement the first functional Agent Core.
6. Introduce the Reason → Act → Observe workflow.

---

## Planned Agent Capabilities

The long-term vision includes several specialized tools.

### Security Analyzer

Potential capabilities:

- Security configuration analysis.
- Identification of suspicious activity.
- Basic security assessment.
- Interpretation of security-related information.

### Log Analyzer

Potential capabilities:

- Log file analysis.
- Error identification.
- Event classification.
- Detection of suspicious patterns.

### Network Monitor

Potential capabilities:

- Network diagnostics.
- Connectivity checks.
- Network information analysis.
- Integration with existing monitoring utilities.

### System Diagnostics

Potential capabilities:

- System information collection.
- Resource utilization analysis.
- Service diagnostics.
- Basic troubleshooting assistance.

### Secure RAG

Provides the agent with access to an authorized technical knowledge base.

These are planned integrations and should not be interpreted as currently available autonomous capabilities.

---

## Project Limitations

WW-P Security AI Agent is an experimental development project.

The current implementation focuses on building and validating its security and AI foundations.

The following capabilities are not yet considered complete:

- Fully autonomous agent execution.
- Complete multi-step planning.
- Persistent agent memory.
- End-to-end Reason-Act-Observe workflow.
- Production-grade human approval mechanisms.
- Complete audit and monitoring infrastructure.
- Comprehensive security validation.

The system should not currently be used as an autonomous production cybersecurity platform.

---

## Author

**Piotr Prochnicki**

IT Support | DevOps | AI & Cybersecurity

GitHub:

https://github.com/PiotrIT2015

Project repository:

https://github.com/PiotrIT2015/secure-rag

Portfolio:

https://piotrit2015.github.io/

---

## Development Philosophy

WW-P Security AI Agent is developed with a security-first approach.

The project explores how local language models, Retrieval-Augmented Generation, controlled tool execution, and agent-based architectures can work together to support cybersecurity and IT operations.

The goal is not simply to build another AI chatbot.

The goal is to develop a modular, explainable, permission-aware AI assistant capable of interacting with technical environments in a controlled and secure manner.

**Build securely. Automate responsibly. Keep humans in control.**
