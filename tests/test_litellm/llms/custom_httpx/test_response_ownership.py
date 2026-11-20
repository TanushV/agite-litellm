import httpx
import pytest
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator


class TrackedStream(httpx.AsyncByteStream):
    def __init__(self):
        self.closes = 0

    async def __aiter__(self):
        yield b'data: {"choices": []}\n\n'

    async def aclose(self):
        self.closes += 1


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
async def test_closing_iterator_releases_response_before_or_after_first_chunk(started):
    stream = TrackedStream()
    response = httpx.Response(200, stream=stream)
    iterator = BaseModelResponseIterator(response.aiter_lines(), sync_stream=False)
    iterator.http_response = response
    if started:
        await iterator.__aiter__().__anext__()
    await iterator.aclose()
    await iterator.aclose()
    assert response.is_closed
    assert stream.closes == 1
