# WW-P Security AI Agent

### Security-focused AI Agent powered by Secure RAG, Local LLMs and Agentic Architecture

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![AI](https://img.shields.io/badge/AI-Agentic%20Systems-purple.svg)
![Security](https://img.shields.io/badge/Security-AI%20Security-red.svg)
![RAG](https://img.shields.io/badge/Architecture-Secure%20RAG-green.svg)
![Status](https://img.shields.io/badge/Status-Active%20Development-orange.svg)

**WW-P Security AI Agent** is a Python-based AI security project designed to evolve from a secure Retrieval-Augmented Generation (RAG) platform into an autonomous, tool-using AI agent.

The system combines document retrieval, access control, encrypted vector storage, input security mechanisms and local Large Language Models.

Its long-term objective is to create a modular AI agent capable of analyzing security-related information, interacting with external tools and supporting technical decision-making while maintaining strict security boundaries.

---

# 1. Project Vision

Traditional AI assistants primarily focus on generating responses.

WW-P Security AI Agent aims to go further.

The project explores how to build an AI system that can:

- Retrieve information from authorized documents.
- Analyze security-related information.
- Detect potentially malicious prompts.
- Identify jailbreak attempts.
- Respect user permissions and access policies.
- Use local language models.
- Interact with registered tools.
- Execute controlled operations.
- Maintain monitoring and security logs.

The central design principle is:

> An AI agent should not only be capable of performing tasks. It should also understand the boundaries within which it is allowed to operate.

---

# 2. Architecture Overview

The architecture is divided into several independent layers.

```text
                USER INTERFACE
                       |
                       v
                AUTHENTICATION
                       |
                       v
                 USER CONTEXT
                       |
                       v
                SECURITY LAYER
                       |
             +---------+---------+
             |                   |
             v                   v
        PROMPT GUARD       JAILBREAK DETECTOR
             |                   |
             +---------+---------+
                       |
                       v
                  AGENT CORE
                       |
             +---------+---------+
             |         |         |
             v         v         v
           SECURE     TOOL      MEMORY
            RAG     REGISTRY    LAYER
             |         |
             v         v
         ENCRYPTED   REGISTERED
           FAISS      TOOLS
             |         |
             +----+----+
                  |
                  v
              LOCAL LLM
                OLLAMA
                  |
                  v
             OUTPUT SECURITY
                  |
                  v
              MONITORING
                  |
                  v
                 ANSWER
```

The architecture is designed to separate decision-making, information retrieval, authorization and tool execution.

---

# 3. Security Architecture

Security is a fundamental part of the project rather than an additional feature.

## Authentication

The authentication subsystem provides:

- User authentication.
- Password hashing using Argon2.
- Session management.
- User context initialization.

## Authorization

The authorization subsystem combines:

- RBAC — Role-Based Access Control.
- ACL — Access Control Lists.
- User roles.
- User groups.
- Document-level permissions.
- Document classification.

A key architectural principle:

**Vector similarity does not grant access to a document.**

FAISS identifies potentially relevant information, while the permission subsystem determines whether the authenticated user is authorized to retrieve it.

## Encrypted Vector Storage

The project uses an encrypted FAISS-based vector storage implementation.

Security properties include:

- AES-256-GCM encryption.
- Data confidentiality.
- Data integrity.
- Encrypted storage of document content and metadata.
- Mapping between document chunks and vector identifiers.

## Prompt Security

The security layer includes:

**PromptGuard**
- Detecting suspicious user instructions.
- Identifying potential prompt injection.
- Applying input security rules.

**JailbreakDetector**
- Detecting attempts to bypass LLM restrictions.
- Identifying suspicious instruction patterns.
- Providing an additional security inspection layer.

**Additional security components**

The architecture also provides modules for:

- PII detection.
- Input and output sanitization.
- Security monitoring.
- Application logging.

---

# 4. Secure RAG Engine

Secure RAG is the information retrieval foundation of the AI agent.

The retrieval pipeline:

```text
             USER QUERY
                  |
                  v
             EMBEDDINGS
                  |
                  v
              FAISS
                  |
                  v
          VECTOR SIMILARITY
                  |
                  v
          CANDIDATE DOCUMENTS
                  |
                  v
             RBAC / ACL
                  |
           +------+------+
           |             |
           v             v
         ALLOW          DENY
           |             |
           v             X
      AUTHORIZED
        CONTEXT
           |
           v
         LLM
           |
           v
        RESPONSE
```

The retrieval subsystem is designed to ensure that only authorized information can enter the LLM context.

This separation is particularly important in systems processing confidential or restricted documents.

---

# 5. Local LLM Integration

The planned language model integration uses **Ollama**.

The objective is to run language models locally instead of relying exclusively on external AI APIs.

Planned capabilities:

- Local LLM inference.
- RAG-based response generation.
- Context-aware responses.
- Integration with the Agent Core.
- Controlled access to registered tools.

Local inference offers additional control over data processing and deployment.

However, local execution alone does not guarantee security. Model access, tool permissions and data handling must also be protected.

---

# 6. Agentic Architecture

The next major development step is transforming Secure RAG into a functional AI agent.

The planned architecture introduces three major components.

## 6.1 Tool Registry

A centralized registry of tools available to the AI agent.

Responsibilities:

- Tool registration.
- Tool discovery.
- Input validation.
- Permission verification.
- Controlled tool execution.
- Tool result handling.

Potential tools:

| Tool | Purpose |
|---|---|
| Secure RAG | Authorized document retrieval |
| Security Analyzer | Security information analysis |
| Log Analyzer | Application and system log analysis |
| Network Monitor | Network information monitoring |
| System Diagnostics | Controlled system diagnostics |

These are planned integration targets, not a claim that every tool is already implemented.

## 6.2 Agent Core

The central decision-making component.

Responsibilities:

- Processing user requests.
- Understanding task objectives.
- Selecting appropriate tools.
- Managing execution workflows.
- Using retrieved context.
- Evaluating tool results.
- Generating responses.

The Agent Core will operate within defined security and authorization boundaries.

## 6.3 Agent Execution Flow

The intended execution model:

```text
             USER REQUEST
                   |
                   v
             INPUT SECURITY
                   |
                   v
               AGENT CORE
                   |
                   v
             TASK ANALYSIS
                   |
                   v
              TOOL SELECTION
                   |
                   v
            PERMISSION CHECK
                   |
             +-----+-----+
             |           |
             v           v
           ALLOW        DENY
             |           |
             v           X
         TOOL EXECUTION
             |
             v
         RESULT ANALYSIS
             |
             v
         CONTEXT / MEMORY
             |
             v
          LLM RESPONSE
             |
             v
        OUTPUT VALIDATION
             |
             v
           MONITORING
             |
             v
            RESULT
```

Tool execution is intended to be controlled rather than unrestricted.

---

# 7. Technology Stack

| Technology | Application |
|---|---|
| Python 3.12+ | Core development |
| FAISS | Vector similarity search |
| Sentence Transformers | Embeddings |
| Ollama | Local LLM inference (planned integration) |
| AES-256-GCM | Data encryption |
| Argon2 | Password hashing |
| SQLite | Local persistence |
| RBAC / ACL | Access control |
| Pytest | Testing |
| Git / GitHub | Version control |

---

# 8. Project Structure

The repository is organized into modular components.

```text
secure-rag/
│
├── authentication/
│   ├── auth.py
│   └── permissions.py
│
├── database/
│
├── documents/
│
├── embeddings/
│   └── embedder.py
│
├── encryption/
│
├── retrieval/
│   └── retriever.py
│
├── security/
│   ├── prompt_guard.py
│   ├── jailbreak_detector.py
│   ├── pii_detector.py
│   └── sanitizer.py
│
├── vectorstore/
│   └── encrypted_faiss.py
│
├── tests/
│
├── app.py
├── config.py
├── requirements.txt
└── README.md
```

The structure will evolve as the Agent Core, Tool Registry and additional components are integrated.

---

# 9. Development Roadmap

Development is divided into incremental stages.

| Stage | Component | Status |
|---|---|---|
| 1 | Secure RAG integration and testing | In progress |
| 2 | Ollama integration | Planned |
| 3 | Tool Registry | Planned |
| 4 | Agent Core | Planned |
| 5 | Agent execution workflow | Planned |
| 6 | Security validation and testing | Planned |
| 7 | Documentation and portfolio release | Planned |

### Stage 1 — Secure RAG Foundation

Objectives:

- Integrate authentication.
- Integrate RBAC/ACL.
- Connect embeddings.
- Integrate encrypted FAISS.
- Implement secure retrieval.
- Verify security mechanisms.
- Complete integration and functional tests.

### Stage 2 — Local LLM

Objectives:

- Integrate Ollama.
- Configure a local language model.
- Connect retrieved context to the LLM.
- Implement response generation.
- Validate model input and output.

### Stage 3 — Tool Registry

Objectives:

- Create a centralized tool registry.
- Define tool interfaces.
- Implement tool registration.
- Validate tool inputs.
- Introduce permission-based execution.

### Stage 4 — Agent Core

Objectives:

- Implement task analysis.
- Introduce tool selection.
- Implement agent orchestration.
- Connect the Agent Core to Secure RAG.
- Integrate Ollama.

### Stage 5 — Agentic Execution

Objectives:

- Implement controlled tool execution.
- Process tool results.
- Introduce multi-step workflows.
- Implement execution limits.
- Add error handling and monitoring.

### Stage 6 — Security Validation

Objectives:

- Test authentication and authorization.
- Test prompt injection protection.
- Test jailbreak detection.
- Test encrypted storage.
- Validate tool permissions.
- Perform integration and regression testing.

---

# 10. Installation

## Requirements

- Python 3.12+
- Git
- Windows or Linux
- Local development environment

Clone the repository:

```bash
git clone https://github.com/PiotrIT2015/secure-rag.git
```

Navigate to the project:

```bash
cd secure-rag
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# 11. Running the Application

Start the application from the project directory:

```bash
python app.py
```

The application currently focuses on the Secure RAG foundation.

Ollama integration and the autonomous agent execution layer are subsequent development stages.

---

# 12. Security Considerations

This project is developed as a security-oriented AI architecture prototype.

Important considerations:

- Never commit API keys or encryption secrets.
- Use environment variables for sensitive configuration.
- Protect encryption keys.
- Validate user input.
- Enforce authorization before exposing retrieved content.
- Restrict tool execution.
- Log security-relevant events.
- Test security boundaries independently.

The project is not currently presented as a production-ready security product.

---

# 13. Project Goals

The long-term objective is to develop a modular AI agent combining:

**Artificial Intelligence**
- Local LLMs.
- Retrieval-Augmented Generation.
- Agentic workflows.
- Tool integration.

**Cybersecurity**
- Authentication.
- Authorization.
- Encrypted data storage.
- Prompt injection detection.
- Jailbreak detection.
- Security monitoring.

**Software Engineering**
- Modular architecture.
- Python development.
- Integration testing.
- Error handling.
- Documentation.

---

# 14. Author

**Piotr Prochnicki**

IT Support & DevOps-oriented IT specialist  
Interested in AI Engineering, Cybersecurity and Agentic Systems.

GitHub: https://github.com/PiotrIT2015

Portfolio: https://piotrit2015.github.io

---

# 15. License

See the LICENSE file in this repository.

---

*WW-P Security AI Agent — Building AI systems with security as a fundamental architectural principle.*

### Secure by Design

Security checks should occur before sensitive information reaches the LLM.

### Explicit Trust Boundaries

External files, retrieved documents and user prompts are treated as untrusted input.

### Fail Closed

When authorization cannot be established, access should be denied rather than granted.

---

# 18. Technology Stack

Current/planned technology stack:

```text
Python 3.12+
    |
    +-- Argon2
    +-- Cryptography
    +-- FAISS
    +-- NumPy
    +-- Embedding model
    +-- LLM
    +-- SQLite
    +-- Tkinter
    +-- YARA
```

Potential local AI infrastructure:

```text
WW-P Secure RAG
       |
       v
Embedding Model
       |
       v
Encrypted FAISS
       |
       v
Local LLM / Ollama
```

This architecture can support local/private deployments where documents should not be sent to an external AI provider.

---

# 19. Project Goals

The long-term objective is to create a modular security-oriented RAG platform that can demonstrate practical knowledge of:

* Python software engineering,
* cybersecurity,
* authentication,
* authorization,
* cryptography,
* secure document processing,
* vector databases,
* embeddings,
* LLMs,
* RAG architecture,
* prompt security,
* data protection,
* monitoring.

The project is intended as a portfolio and engineering-learning project rather than a claim of production-grade certification or security assurance.

---

# 20. Security Disclaimer

WW-P Secure RAG is a development and portfolio project.

It should **not** be assumed to be secure for production deployment solely because individual security mechanisms are implemented.

Before production use, the system would require, among other things:

* threat modeling,
* dependency auditing,
* penetration testing,
* secure secret management,
* key rotation,
* hardened deployment,
* access-control review,
* logging and audit review,
* fuzz testing,
* vulnerability scanning,
* secure configuration management,
* independent security assessment.

Security controls should be independently validated before handling sensitive production information.

---

# 21. Recommended Development Workflow

Development should proceed in small, testable stages:

```text
1. Authentication
       |
       v
2. Authorization
       |
       v
3. Document Processing
       |
       v
4. Embeddings
       |
       v
5. Encrypted Vector Store
       |
       v
6. Secure Retrieval
       |
       v
7. Prompt Security
       |
       v
8. LLM
       |
       v
9. Output Security
       |
       v
10. Monitoring
       |
       v
11. GUI
```

Each layer should be tested independently before integrating it with the next one.

---

# 22. Example Final Architecture

The intended final architecture is:

```text
┌─────────────────────────────────────────────────────────────┐
│                    WW-P SECURE RAG                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐                                           │
│  │     GUI      │                                           │
│  │   Tkinter    │                                           │
│  └──────┬───────┘                                           │
│         │                                                   │
│         v                                                   │
│  ┌──────────────┐                                           │
│  │     APP      │                                           │
│  │  Orchestrator│                                          │
│  └──────┬───────┘                                           │
│         │                                                   │
│    ┌────┴───────────────────────────┐                       │
│    │                                │                       │
│    v                                v                       │
│ Authentication                 Security                     │
│    │                         ┌──────────────┐                │
│    v                         │ Prompt Guard │                │
│ UserContext                  │ Jailbreak    │                │
│    │                         │ PII          │                │
│    v                         │ Sanitizer    │                │
│ RBAC / ACL                   └──────────────┘                │
│    │                                                        │
│    └───────────────┐                                        │
│                    v                                        │
│             Secure Retriever                                │
│                    │                                        │
│           ┌────────┴────────┐                               │
│           v                 v                               │
│      Embeddings       Encrypted FAISS                       │
│                             │                               │
│                             v                               │
│                       Authorized                            │
│                         Context                             │
│                             │                               │
│                             v                               │
│                            LLM                              │
│                             │                               │
│                             v                               │
│                         Sanitizer                           │
│                             │                               │
│                             v                               │
│                           User                              │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ Monitoring │ Logging │ Metrics │ SQLite │ Cryptography      │
└─────────────────────────────────────────────────────────────┘
```

---

# 23. Status

Current development status:

| Component               | Status |
| ----------------------- | ------ |
| Project structure       | ✅      |
| `app.py`                | ✅      |
| Configuration           | 🟡     |
| Authentication          | ✅      |
| Argon2 password hashing | ✅      |
| UserContext             | ✅      |
| RBAC                    | 🟡     |
| ACL                     | 🟡     |
| Document loader         | 🟡     |
| Document extractor      | 🟡     |
| Chunking                | 🟡     |
| Embeddings              | 🟡     |
| Encrypted FAISS         | 🟡     |
| Secure Retriever        | ✅      |
| Prompt Guard            | 🟡     |
| Jailbreak Detector      | ✅     |
| PII Detector            | ✅     |
| Sanitizer               | ✅     |
| LLM integration         | 🟡     |
| Monitoring              | 🟡     |
| SQLite persistence      | 🟡     |
| Tkinter GUI             | 🟡     |
| End-to-end integration  | 🔜     |
| Security testing        | 🔜     |

Legend:

```text
✅ implemented
🟡 in development / integration
🔜 planned
```

---

# 24. Conclusion

WW-P Secure RAG is an attempt to treat RAG not merely as an AI application, but as a **security-sensitive information system**.

The fundamental architectural distinction is:

```text
Semantic relevance ≠ authorization
```

A document can be highly relevant to a user's query and still be inaccessible to that user.

Therefore:

```text
Vector Search
     ≠
Access Control
```

The Secure RAG architecture places authorization between retrieval and LLM generation:

```text
             Vector Search
                   |
                   v
             Candidates
                   |
                   v
             RBAC / ACL
                   |
                   v
         Authorized Context
                   |
                   v
                  LLM
```

This principle forms the foundation of the WW-P Secure RAG security architecture.

---

## License

License: Proprietary — All Rights Reserved
This software is proprietary and confidential. No copying, modification, distribution, redistribution, 
or commercial use is permitted without prior written authorization from the copyright holder. 
Licensing inquiries: piotr.prochnicki@gmail.com

---

## Author

**Piotr Próchnicki**

WW-P Secure RAG — Python / Cybersecurity / AI Security Portfolio Project.
