"""Real cache and HTTP transport contracts; no patched dependency methods."""

import base64
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import stat
import time

import httpx
import pytest
from filelock import FileLock

from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.llms.chatgpt.common_utils import GetAccessTokenError, RefreshAccessTokenError


def jwt(**claims):
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


def tokens(account="test-account", refresh="rotated-refresh"):
    return {
        "access_token": jwt(exp=int(time.time()) + 3600),
        "refresh_token": refresh,
        "id_token": jwt(**{"https://api.openai.com/auth": {"chatgpt_account_id": account}}),
    }


def seed(directory, *, expires=None, refresh="old-refresh", filename="auth.json"):
    directory.mkdir(parents=True, exist_ok=True)
    record = dict(tokens(), expires_at=expires or time.time() - 1, refresh_token=refresh)
    path = directory / filename
    path.write_text(json.dumps(record))
    return path


def auth(directory, handler, **kwargs):
    return Authenticator(
        token_dir=str(directory), non_interactive=True,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs,
    )


def no_network(request):
    pytest.fail(f"Unexpected authentication request: {request.url.path}")


def test_valid_cache_is_private_and_uses_no_network(tmp_path):
    cache = seed(tmp_path / "cache", expires=time.time() + 3600)
    client = auth(cache.parent, no_network)
    assert client.get_access_token() == json.loads(cache.read_text())["access_token"]
    assert stat.S_IMODE(cache.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(cache.stat().st_mode) == 0o600
    assert stat.S_IMODE(Path(str(cache) + ".lock").stat().st_mode) == 0o600


@pytest.mark.parametrize("remaining", [-100, 30])
def test_refresh_is_persisted_before_return_and_reused_after_restart(tmp_path, remaining):
    cache = seed(tmp_path / "cache", expires=time.time() + remaining)
    response = tokens("second-account")
    requests = []

    def refresh(request):
        requests.append(request)
        assert request.url.path == "/oauth/token"
        assert json.loads(request.content)["refresh_token"] == "old-refresh"
        return httpx.Response(200, json=response)

    client = auth(cache.parent, refresh)
    assert client.get_access_token() == response["access_token"]
    saved = json.loads(cache.read_text())
    assert saved["refresh_token"] == "rotated-refresh"
    assert saved["account_id"] == "second-account"
    assert stat.S_IMODE(cache.stat().st_mode) == 0o600
    restarted = auth(cache.parent, no_network)
    assert restarted.get_access_token() == response["access_token"]
    assert restarted.get_account_id() == "second-account"
    assert len(requests) == 1


@pytest.mark.parametrize("contents", [None, "not json", "[]", "{}", '{"expires_at":"broken"}'])
def test_missing_or_malformed_cache_fails_without_interactive_login(tmp_path, contents):
    directory = tmp_path / "cache"
    directory.mkdir()
    if contents is not None:
        (directory / "auth.json").write_text(contents)
    with pytest.raises(GetAccessTokenError, match="interactive login is disabled"):
        auth(directory, no_network).get_access_token()


@pytest.mark.parametrize("status,body", [(401, {"error": "secret-marker"}), (200, {"access_token": "secret-marker"})])
def test_refresh_failure_is_redacted_and_does_not_fall_back(tmp_path, status, body):
    cache = seed(tmp_path / "cache")
    before = cache.read_bytes()
    requests = []

    def failure(request):
        requests.append(request)
        return httpx.Response(status, json=body)

    with pytest.raises(RefreshAccessTokenError) as error:
        auth(cache.parent, failure).get_access_token()
    assert "secret-marker" not in str(error.value)
    assert "old-refresh" not in str(error.value)
    assert cache.read_bytes() == before
    assert [r.url.path for r in requests] == ["/oauth/token"]


def test_failed_persistence_never_returns_fresh_token_or_destroys_old_cache(tmp_path):
    cache = seed(tmp_path / "cache")
    before = cache.read_bytes()

    def refresh(request):
        cache.parent.chmod(0o500)
        return httpx.Response(200, json=tokens())

    try:
        with pytest.raises(GetAccessTokenError, match="persist"):
            auth(cache.parent, refresh).get_access_token()
        assert cache.read_bytes() == before
    finally:
        cache.parent.chmod(0o700)


def test_parallel_threads_recheck_cache_under_shared_lock(tmp_path):
    cache = seed(tmp_path / "cache")
    calls = []

    def refresh(request):
        calls.append(1)
        time.sleep(0.05)
        return httpx.Response(200, json=tokens())

    clients = [auth(cache.parent, refresh) for _ in range(8)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda client: client.get_access_token(), clients))
    assert len(set(results)) == 1
    assert len(calls) == 1
    assert json.loads(cache.read_text())["refresh_token"] == "rotated-refresh"


def process_refresh(directory, counter):
    def refresh(request):
        with open(counter, "a") as stream:
            stream.write("refresh\n")
        time.sleep(0.1)
        return httpx.Response(200, json=tokens())
    auth(Path(directory), refresh).get_access_token()


@pytest.mark.skipif(os.name != "posix", reason="POSIX deployment process contract")
def test_separate_processes_refresh_only_once(tmp_path):
    cache = seed(tmp_path / "cache")
    counter = tmp_path / "requests"
    context = multiprocessing.get_context("fork")
    processes = [context.Process(target=process_refresh, args=(str(cache.parent), str(counter))) for _ in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert counter.read_text().splitlines() == ["refresh"]


def test_lock_contention_has_a_bounded_failure(tmp_path):
    cache = seed(tmp_path / "cache")
    client = auth(cache.parent, no_network, lock_timeout=0.01)
    with FileLock(str(cache) + ".lock"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(client.get_access_token)
            with pytest.raises(GetAccessTokenError, match="lock"):
                future.result(timeout=2)


def test_explicit_filename_selects_one_cache(tmp_path):
    directory = tmp_path / "cache"
    selected = seed(directory, expires=time.time() + 3600, filename="selected.json")
    (directory / "auth.json").write_text("{}")
    client = auth(directory, no_network, auth_file="selected.json")
    assert client.get_access_token() == json.loads(selected.read_text())["access_token"]


@pytest.mark.parametrize("filename", ["../outside.json", "/tmp/outside.json"])
def test_cache_filename_cannot_escape_private_directory(tmp_path, filename):
    with pytest.raises(ValueError, match="filename"):
        auth(tmp_path / "cache", no_network, auth_file=filename)
