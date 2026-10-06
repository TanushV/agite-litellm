# Maintained credential and HTTP stream lifecycle

This is an explicit MIT-licensed fork of BerriAI/LiteLLM. Version
`1.88.6+agite.1` is based on `v1.88.6`
(`b504a0eeb6f8a26daae3a2ebca12ccc2cacd3a2f`) and retains the maintained headless
ChatGPT credential and HTTP response-ownership fixes below. `filelock` remains
a direct dependency. All upstream attribution remains.

## Operational contract

Set `CHATGPT_NON_INTERACTIVE=true` in unattended processes. Keep one dedicated
session per deployed cache; do not copy a desktop session that another machine
will continue refreshing. Revocation requires explicit operator reauthorization.
This fork is not an OpenAI-supported authentication integration.

`CHATGPT_TOKEN_DIR` selects a dedicated private directory; `CHATGPT_AUTH_FILE`
selects one filename inside it (default `auth.json`). The directory must be on a
persistent local filesystem and writable by the process. The authenticator
enforces directory mode 0700 and file/lock mode 0600. It serializes reads,
refreshes and writes across processes, with a 60-second lock timeout and a
30-second refresh HTTP timeout. Refreshed tokens are flushed and atomically
replaced before use. Disk failure is an authentication failure, not permission
to continue with an unpersisted rotating credential.

The native Responses transport still constructs this canonical authenticator.
There is no alternate agent loop or runtime patch. Explicit constructor options
for directory, filename, interactivity, HTTP client and lock timeout support
embedded use and isolated transport tests. All credentials remain in the
existing LiteLLM flat JSON format; no desktop-cache converter is included.

## Verification and maintenance

Run `python -m pytest -q tests/test_litellm/llms/chatgpt`. These tests use
disposable credentials, the real filesystem, subprocesses and HTTP transports;
they do not patch production methods. Audit both the fork's complete dependency
resolution and upstream `litellm==1.88.6`: PyPI advisory lookup does not recognize
local fork versions or URL requirements. A skipped fork is not an audit pass.
Rebase only after rerunning lifecycle and native Responses regression tests.

Build with `python -m pip wheel --no-deps .`. Install a released wheel using its
SHA256 hash, never a moving Git branch or a startup edit of site-packages.

## 1.84.0+agite.2: streamed response ownership

The async HTTP chat transport transfers ownership of the open HTTP response to
its base model iterator. `aclose()` closes both the line iterator and the response,
including when no chunk was consumed. Construction failures close the response
before propagating the error. The outer CustomStreamWrapper already delegates
closure to the inner iterator; callers must close streams in a `finally` block.

This fixes early timeout and cancellation leaving connections acquired on the
OpenRouter HTTP transport. It does not introduce retries, alter provider routing,
close clients during cache eviction or replace the transport. The shipped delta
from agite.1 is limited to `base_model_iterator.py` and `llm_http_handler.py`.

Run `tests/test_litellm/llms/custom_httpx/test_response_ownership.py` for closure
before and after the first chunk; the runtime also tests actual loopback HTTP
streams, cancellation, inactivity and observed versus synthetic status codes.

## 1.88.6+agite.1: upstream security baseline

Rebased the maintained credential and response-ownership changes onto upstream
`v1.88.6` (`b504a0eeb6f8a26daae3a2ebca12ccc2cacd3a2f`). This includes the upstream
fix for GHSA-3cv6-jpf6-8222 (PYSEC-2026-4066), covering unvalidated proxy routing
and credential parameters. The prior 1.84 sections above record historical
patch provenance; this release must be audited as upstream `litellm==1.88.6`.
The upstream Python support bound is preserved: Python 3.10 through 3.13.

Source: https://github.com/BerriAI/litellm/security/advisories/GHSA-3cv6-jpf6-8222
