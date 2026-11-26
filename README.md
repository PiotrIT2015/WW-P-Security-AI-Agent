# WW-P Security AI Agent

## Table of contents
* [General info](#general-info)
* [Running](#running)
* [Technologies](#technologies)

## Running

```
python app.py
Secure RAG > Jak działa monitoring bezpieczeństwa?
```


## General info

**Security First. Local AI. Controlled Autonomy.**

WW-P Security AI Agent is a local-first AI agent project focused on
cybersecurity and system-administration workflows. It builds on the
Secure RAG foundation and combines document retrieval, access control,
local LLM inference, controlled tool execution, and defensive
prompt-security mechanisms.

The project is developed as a practical portfolio implementation
exploring the intersection of **AI Engineering, Cybersecurity, and
DevOps**.

> **Project status (26 November 2026):** Core components and the current
> automated test suite are implemented. The complete test suite reports
> **491 passed**. The project is best described as a functionally
> advanced MVP / development release. Production readiness has not yet
> been established through full deployment and live-environment
> validation.

## Table of contents

-   [Project goals](#project-goals)
-   [Key capabilities](#key-capabilities)
-   [Architecture](#architecture)
-   [Technology stack](#technology-stack)
-   [Implementation status](#implementation-status)
-   [Automated tests](#automated-tests)
-   [Development roadmap](#development-roadmap)
-   [Environment and execution](#environment-and-execution)
-   [Security notes and limitations](#security-notes-and-limitations)

## Project goals

The project aims to provide a controlled local AI assistant that can:

-   retrieve information from technical and cybersecurity documents;
-   respect authentication, roles, and document-level access rules;
-   use a locally hosted language model through Ollama;
-   invoke registered tools through a controlled execution layer;
-   preserve source attribution and distinguish trusted instructions
    from retrieved content;
-   handle errors and insufficient evidence safely.

The design prioritizes local processing, least privilege, explicit tool
permissions, and auditable execution over unrestricted autonomy.

## Key capabilities

### Secure RAG foundation

-   Authentication and role-based access control (RBAC).
-   Document-level access control (ACL).
-   Document ingestion and processing pipeline.
-   Text embeddings using `sentence-transformers/all-MiniLM-L6-v2` (384
    dimensions).
-   FAISS vector retrieval.
-   Encrypted vector-store persistence using AES-256-GCM.
-   Authorized retrieval and retrieval-context validation.

### Prompt and input security

-   PromptGuard input checks.
-   JailbreakDetector checks.
-   PII detection and input sanitization components.
-   Retrieved document content is treated as untrusted data, not as
    system instructions.
-   RAG context is serialized and bounded before it is added to model
    messages.

### Local LLM and agent execution

-   Ollama integration for local model inference.
-   Response generation and response validation.
-   Tool Registry and controlled tool execution.
-   Tool permissions, call parsing, and execution bridge.
-   Agent state and execution-loop components.
-   RAG context and source metadata passed into response generation.

## Architecture

High-level logical flow:

``` text
User Request
    |
    v
Authentication / User Context
    |
    v
Input Security
(PromptGuard / JailbreakDetector)
    |
    v
Agent Core
    |                         |
    |                         v
    |                    Tool Registry
    |                         |
    |                    Permission Checks
    |                         |
    |                    Tool Execution Bridge
    |
    v
Secure RAG Adapter
    |
    v
Authorized Retrieval
(RBAC / ACL / Secure Retriever)
    |
    v
RAG Context Builder
    |
    v
Response Generator <----> Ollama
    |
    v
Validated Response + Source Attribution
```

This is a logical overview; individual modules and runtime wiring may
evolve as the project develops.

## Technology stack

  ------------------------------------------------------------------------------
  Area                                Technology / component
  ----------------------------------- ------------------------------------------
  Language                            Python

  Runtime tested during development   Python 3.10.6

  Local LLM runtime                   Ollama

  LLM                                 Qwen-family local model; development
                                      configuration included a 3.1B parameter
                                      Q4_K_M model

  Embeddings                          Sentence Transformers

  Embedding model                     `sentence-transformers/all-MiniLM-L6-v2`

  Vector search                       FAISS

  Vector-store encryption             AES-256-GCM

  Testing                             pytest

  Platform                            Windows development environment;
                                      Linux-oriented deployment is a possible
                                      future target
  ------------------------------------------------------------------------------

## Implementation status

Status is recorded as of **26 November 2026**.

  -----------------------------------------------------------------------------
  Stage             Scope             Status            Notes
  ----------------- ----------------- ----------------- -----------------------
  1                 Secure RAG        **Implemented**   Authentication,
                    foundation                          RBAC/ACL, document
                                                        processing, embeddings,
                                                        encrypted FAISS
                                                        storage, secure
                                                        retrieval, and
                                                        retrieval validation
                                                        are present.

  2                 Local LLM         **Implemented**   Ollama client and
                    integration                         response-generation
                                                        components are
                                                        integrated.

  3                 Tool Registry and **Implemented**   Registry, executor,
                    execution                           permissions, tool-call
                    infrastructure                      parser, and execution
                                                        bridge have dedicated
                                                        tests.

  4                 Agent Core        **Implemented /   Agent state, core
                                      tested**          execution flow, context
                                                        preparation, and error
                                                        handling have been
                                                        developed and tested.

  5                 Controlled agent  **Implemented in  The project includes an
                    execution         its current       execution loop and
                                      scope**           controlled tool-call
                                                        flow. More advanced
                                                        autonomous planning
                                                        remains future work.

  6                 Response and      **Core test work  The response-quality,
                    security          completed**       hallucination,
                    validation                          insufficient-context,
                                                        citation, and
                                                        prompt-injection test
                                                        milestones are
                                                        complete. See the
                                                        detailed status below.

  7                 Documentation and **In progress**   This README documents
                    release readiness                   the current
                                                        implementation. Full
                                                        deployment and
                                                        operational validation
                                                        remain outstanding.
  -----------------------------------------------------------------------------

### Stage 6 -- detailed test milestones

  -----------------------------------------------------------------------
  Sub-stage               Name                    Status
  ----------------------- ----------------------- -----------------------
  3.6.1                   Response Quality Tests  **Completed**

  3.6.2                   Hallucination Detection **Completed**

  3.6.3                   Insufficient Context    **Completed**
                          Handling                

  3.6.4                   Citation Correctness    **Completed**
                          Tests                   

  3.6.5                   Prompt Injection        **Completed**
                          Response Tests          

  3.6.6                   Cross-User Response     **Deferred**
                          Security                

  3.6.7                   Response Generator      **Skipped / not
                          Integration Tests       implemented as a
                                                  separate milestone**
  -----------------------------------------------------------------------

**Why 3.6.6 is deferred:** the current project scope is a single-user
local agent rather than a multi-user service. Existing identity
validation, access-control, and RAG authorization tests remain relevant;
dedicated cross-user response-isolation tests can be added if multi-user
support is introduced.

**Why 3.6.7 is skipped:** much of the generator and RAG behavior is
already covered by component, RAG, response-quality, citation, and
prompt-injection tests. A separate integration-test milestone was judged
to have lower immediate value. This is a documented scope decision, not
a claim that every possible end-to-end integration scenario has been
tested.

## Automated tests

Latest recorded full-suite result:

``` text
491 passed
```

Run the complete suite from the repository root:

``` powershell
python -m pytest -v
```

Run selected response and security tests:

``` powershell
python -m pytest tests/test_agent_response_quality.py -v
python -m pytest tests/test_agent_citation_correctness.py -v
python -m pytest tests/test_agent_prompt_injection.py -v
```

The reported 491-pass result is the latest recorded result for this
development state. Re-run the suite after pulling changes or modifying
the environment to verify the current checkout.

## Development roadmap

The next work should focus on validating the complete application rather
than adding test milestones solely for completeness:

1.  Run the agent against the configured local Ollama model and verify
    real request/response behavior.
2.  Exercise representative Secure RAG queries using actual indexed
    documents and access rules.
3.  Verify tool execution end to end, including permission denials and
    failure paths.
4.  Review secrets, environment configuration, logging, and
    encrypted-store key handling.
5.  Document reproducible installation, configuration, and startup steps
    for a clean environment.
6.  Perform deployment-oriented checks before describing the project as
    production-ready.

Potential future extensions include richer planning, persistent agent
memory, evaluation of tool outcomes, Human-in-the-Loop approval for
sensitive actions, and multi-user isolation if the product scope
changes.

## Environment and execution

The project has been developed and tested in a Windows environment using
Python 3.10.6 and a virtual environment.

Typical development setup:

``` powershell
# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install project dependencies
python -m pip install -r requirements.txt

# Run automated tests
python -m pytest -v
```

Ollama must be installed and running separately, and the configured
model must be available locally. Set the model name and other required
settings according to the project's current configuration and
environment template. Do not commit API keys, encryption keys, JWT
secrets, or populated `.env` files to version control.

The exact application startup command and environment-variable names
should be taken from the current repository configuration; they may
change during development.

## Security notes and limitations

-   This is a development / portfolio project, not a certified security
    product.
-   Passing automated tests does not prove the absence of
    vulnerabilities.
-   The recorded test suite does not replace live validation with the
    actual Ollama runtime, real documents, and the intended deployment
    environment.
-   Encryption depends on correct key generation, storage, rotation, and
    recovery procedures.
-   Tool permissions should follow least privilege. Sensitive or
    destructive actions should require explicit human approval before
    execution.
-   Do not expose the local agent or its tool-execution interface to
    untrusted networks without a separate security review.
-   Multi-user response isolation is not a current product requirement;
    it is deferred until multi-user functionality is introduced.

## Repository

-   GitHub: https://github.com/PiotrIT2015/secure-rag

------------------------------------------------------------------------

**WW-P Security AI Agent --- Security First. Local AI. Controlled Autonomy.**

## Technologies
* Python version: 3.10
* ollama
* qwen2


