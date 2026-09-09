# Maintained ChatGPT credential lifecycle

This is an explicit MIT-licensed fork of BerriAI/LiteLLM, based on `v1.84.0`
(`e1fc955464bf493c15aef08f98e0e22bdf24d4cf`). The baseline's 1,797 shipped Python
files match the PyPI 1.84.0 wheel. Version `1.84.0+agite.1` changes only
`litellm/llms/chatgpt/authenticator.py` in the shipped package and declares
`filelock` as a direct dependency. All upstream attribution remains.

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
resolution and upstream `litellm==1.84.0`: PyPI advisory lookup does not recognize
local fork versions or URL requirements. A skipped fork is not an audit pass.
Rebase only after rerunning lifecycle and native Responses regression tests.

Build with `python -m pip wheel --no-deps .`. Install a released wheel using its
SHA256 hash, never a moving Git branch or a startup edit of site-packages.
