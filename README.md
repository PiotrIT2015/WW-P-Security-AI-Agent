# WW-P Security AI Agent

**Status:** Active development — integration and validation phase  
**Project:** Secure, tool-enabled AI agent built on top of the WW-P Secure RAG foundation  
**Primary stack:** Python 3.10, Ollama, Qwen, FAISS, encrypted vector storage, pytest

> WW-P Security AI Agent is a security-oriented AI agent project that combines retrieval-augmented generation (RAG), access control, local LLM inference and controlled tool execution. The project is under active development. Individual components have been implemented and tested, but the complete end-to-end system has not yet been declared finished or production-ready.

---

## 1. Project overview

The goal of **WW-P Security AI Agent** is to build a local, security-conscious AI agent capable of:

- answering questions using authorized internal knowledge,
- retrieving information through a Secure RAG pipeline,
- communicating with a locally hosted language model through Ollama,
- selecting and executing registered tools through a controlled execution layer,
- maintaining agent execution state and conversation context,
- applying security controls before information is retrieved or actions are executed.

The project builds on the **WW-P Secure RAG** foundation and extends it with agent orchestration and tool-use capabilities.

## 2. High-level architecture

```text
                         USER
                           |
                           v
                 WW-P SECURITY AI AGENT
                           |
                           v
                       AGENT CORE
                           |
                +----------+----------+
                |                     |
                v                     v
           SECURE RAG             TOOL SYSTEM
                |                     |
        +-------+-------+       +-----+------+
        |               |       |            |
        v               v       v            v
   RETRIEVER       ACCESS       TOOL       PERMISSIONS
                   CONTROL      REGISTRY
        |                           |
        v                           v
 AUTHORIZED CONTEXT            TOOL EXECUTOR
        |                           |
        +-------------+-------------+
                      |
                      v
                 OLLAMA / QWEN
                      |
                      v
             RESPONSE GENERATION
                      |
                      v
                FINAL RESPONSE
```

The intended agent execution cycle is:

```text
REASON -> ACT -> OBSERVE -> EVALUATE -> RESPOND
```

The complete cycle, including all integrations and security boundaries, still requires end-to-end verification.

## 3. Implemented components and current status

The table below reflects development progress known as of **26 November 2026**. “Implemented” does not automatically mean that the component has passed full-system or production-level validation.

| Component | Current status | Notes |
|---|---|---|
| Authentication | Implemented | Authentication foundation is present. |
| RBAC / ACL | Implemented | Role- and access-control mechanisms are present. |
| Document processing | Implemented | Secure RAG document pipeline. |
| Embeddings | Implemented | `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, normalized embeddings. |
| Encrypted vector storage | Implemented | FAISS storage protected with AES-256-GCM. |
| Secure retrieval | Implemented | Retrieval is integrated with access-control checks. |
| PromptGuard | Implemented | Prompt security rules are present. |
| Jailbreak detection | Implemented | Detection rules are present. |
| Ollama client | Implemented | Local LLM communication layer. |
| Qwen integration | In progress / configured | Qwen model has been prepared in the local Ollama environment; full application flow still needs validation. |
| Tool Registry | Implemented | Tool registration layer. |
| Tool Executor | Implemented | Controlled tool execution component. |
| Tool permissions | Implemented | Permission checks for tool execution. |
| Tool Call Parser | Implemented | Parses model tool-call requests. |
| Tool Execution Bridge | Implemented | Connects agent tool requests with execution. |
| Agent state and context | Implemented / tested | Agent state and context preparation are present. |
| Agent Core | In integration | Core orchestration is being refined and tested. |
| Agent execution loop | Implemented / tested | Iteration and tool-execution flow has dedicated tests. |
| RAG Adapter | Implemented / tested | Dedicated adapter tests have passed. |
| RAG Context Builder | Implemented / tested | Builds the context supplied to response generation. |
| RAG integration | In validation | Component-level tests have passed; full response flow is still being checked. |
| Response Generator | In validation | Latest reported test run: 5 passed and 12 failed out of 17. |
| RAG response generation | In validation | Additional integration and response-generation tests are in progress. |
| Conversation memory | Planned | Persistent or structured conversational memory remains to be completed and verified. |
| Planning | Planned | Multi-step task planning. |
| Evaluator | Planned | Evaluation of generated answers and agent actions. |
| Human-in-the-Loop (HITL) | Planned | User approval for sensitive or consequential operations. |
| End-to-end security validation | Pending | Requires integrated security and end-to-end testing. |

## 4. Testing progress

The project has a growing automated test suite covering the main building blocks.

Previously reported successful test results include:

| Test area | Reported result |
|---|---:|
| Tool Call Parser | 42 passed |
| Tool Execution Bridge | 28 passed |
| RAG Adapter | 17 passed |
| RAG Context Builder | 13 passed |
| Agent Core | 10 passed in an earlier test run |
| Agent Execution Loop | 13 passed |
| Earlier combined regression run | 219 passed |

**Important:** These are results from separate test runs at earlier development points. They must not be added together or interpreted as the current total number of passing tests. They also do not prove that the complete application works end to end.

### Latest known issue — 26 November 2026

The latest reported run of:

```powershell
python -m pytest tests/test_agent_response_generator.py -v
```

reported:

```text
12 failed, 5 passed
```

The failures indicate that response-generator validation and RAG-context validation still need attention. In particular, the current validation behavior does not yet match all test expectations, and a valid sample RAG context was rejected because of a source-count validation condition.

The next step is to resolve these failures, rerun the focused tests, and then run the wider regression suite.

## 5. Security foundations

The project is designed around security controls at multiple layers:

- **Authentication:** identify the user interacting with the system.
- **RBAC / ACL:** restrict access according to roles and document permissions.
- **Secure retrieval:** return only information the requesting user is authorized to access.
- **Encrypted vector storage:** protect the FAISS index using AES-256-GCM.
- **Prompt protection:** apply PromptGuard and jailbreak-detection checks.
- **Tool permissions:** restrict which tools can be invoked.
- **Controlled execution:** route tool calls through the registry, executor and execution bridge.

These controls are implemented as project components, but their effectiveness must be verified through integrated negative tests, adversarial tests and security review. The project should not be described as production-secure until that work is complete.

## 6. Technology stack

| Area | Technology |
|---|---|
| Programming language | Python 3.10 |
| Local LLM runtime | Ollama |
| Language model | Qwen (local model configuration) |
| Embeddings | Sentence Transformers |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector search | FAISS |
| Vector index encryption | AES-256-GCM |
| Testing | pytest |
| Operating environment | Windows 11, Python virtual environment |

Exact dependency versions and setup requirements should be taken from the repository's dependency files and environment configuration.

## 7. Development roadmap

The immediate priority is to finish the current integration work before adding more advanced agent features.

### Priority 1 — Finish the current integration

- [ ] Fix `agent/response_generator.py` validation issues.
- [ ] Make RAG context validation consistent with the RAG context builder and source-attribution model.
- [ ] Pass `tests/test_agent_response_generator.py`.
- [ ] Pass the RAG response-generation tests.
- [ ] Verify the integration between Agent Core, RAG Adapter, RAG Context Builder and Response Generator.
- [ ] Run the full regression suite and record the current results.

### Priority 2 — Verify real agent behavior

- [ ] Run the complete application with Ollama and the configured Qwen model.
- [ ] Verify a normal user question and a RAG-grounded answer.
- [ ] Verify that unauthorized documents are not disclosed.
- [ ] Verify tool selection, permission checks and tool execution.
- [ ] Verify iteration limits and error handling.
- [ ] Verify the complete Reason → Act → Observe cycle.

### Priority 3 — Extend agent capabilities

- [ ] Conversation memory.
- [ ] Multi-step planning.
- [ ] Evaluator for responses and actions.
- [ ] Human-in-the-Loop approval for sensitive actions.

### Priority 4 — Release readiness

- [ ] End-to-end tests.
- [ ] Security and adversarial tests.
- [ ] Configuration and secrets review.
- [ ] Logging and error-handling review.
- [ ] Installation and usage documentation.
- [ ] Final release checklist and versioning.

## 8. Current project maturity

At the current stage, the project can be described as:

- **Secure RAG foundation:** substantially implemented.
- **Tool system:** implemented at component level.
- **Agent orchestration:** under active integration and validation.
- **Complete autonomous agent workflow:** not yet fully verified.
- **Production readiness:** not established.

The project is therefore an **advanced development prototype**, not yet a completed production system.

## 9. Development environment

The project has been developed and tested in a Windows environment using a Python virtual environment.

Example test command:

```powershell
python -m pytest -v
```

Run focused tests while developing individual components, for example:

```powershell
python -m pytest tests/test_agent_response_generator.py -v
```

Use the repository's actual configuration files for installation, environment variables, model setup and application startup. Do not commit `.env` files, API keys, encryption keys, JWT secrets or other credentials to version control.

## 10. Project goals

The long-term goal is to deliver a modular, local-first AI security agent that combines:

1. Secure retrieval of authorized knowledge.
2. Local LLM-based reasoning and response generation.
3. Permission-controlled tool execution.
4. Observable and bounded agent execution.
5. Evaluation and human approval where appropriate.

The guiding principle is that **agent capability must not bypass the security model**. Retrieval, context construction and tool execution must preserve authorization boundaries.

---

**Project:** WW-P Security AI Agent  
**Repository:** https://github.com/PiotrIT2015/secure-rag  
**Status date:** 26 November 2026  
**Development status:** Active — Agent Core / RAG / Response Generator integration

| Etap | Status |
|---|---|
| Secure RAG – podstawowa infrastruktura | ✅ |
| Tool Registry i Tool Executor | ✅ |
| Tool Call Parser i Execution Bridge | ✅ |
| Agent Core i Execution Loop | ✅ |
| RAG Adapter i Context Builder | ✅ |
| RAG Integration i E2E | ✅ |
| Etap | Nazwa | Status |
|---|---|---|
| 3.6.1 | Response Quality Tests | ✅ |
| 3.6.2 | Hallucination Detection | ✅ |
| 3.6.3 | Insufficient Context Handling | ✅ |
| 3.6.4 | Citation Correctness Tests | ✅ |
| 3.6.5 | Prompt Injection Response Tests | ⏳ |
| 3.6.6 | Cross-User Response Security | ⏳ |
| 3.6.7 | Response Generator Integration Tests | ⏳ |