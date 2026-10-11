import json

import pytest

from litellm.completion_extras.litellm_responses_transformation.transformation import (
    OpenAiResponsesToChatCompletionStreamIterator,
)


def item_event(kind, index, arguments=""):
    return {
        "type": f"response.output_item.{kind}",
        "output_index": index,
        "item": {
            "type": "function_call",
            "id": f"fc_{index}",
            "call_id": f"call_{index}",
            "name": f"operation_{index}",
            "arguments": arguments,
        },
    }


def arguments_event(kind, index, value):
    return {
        "type": f"response.function_call_arguments.{kind}",
        "output_index": index,
        "item_id": f"fc_{index}",
        "delta" if kind == "delta" else "arguments": value,
    }


def collect(events):
    iterator = OpenAiResponsesToChatCompletionStreamIterator(iter(()), sync_stream=True)
    calls = {}
    for event in events:
        chunk = iterator.chunk_parser(event)
        for choice in chunk.choices:
            for call in choice.delta.tool_calls or []:
                target = calls.setdefault(call.index, {"id": "", "name": "", "arguments": ""})
                target["id"] += call.id or ""
                target["name"] += call.function.name or ""
                target["arguments"] += call.function.arguments or ""
            assert choice.finish_reason is None
    return calls


@pytest.mark.parametrize("prefix", ["", '{"path":', '{"path":"memo.json"}'])
def test_should_reconcile_complete_arguments_without_repeating_streamed_prefix(prefix):
    arguments = '{"path":"memo.json"}'
    events = [item_event("added", 2)]
    if prefix:
        events.append(arguments_event("delta", 2, prefix))
    events += [arguments_event("done", 2, arguments), item_event("done", 2, arguments)]
    assert collect(events) == {
        2: {"id": "call_2", "name": "operation_2", "arguments": arguments}
    }


def test_should_preserve_interleaved_function_calls_and_exact_unicode_arguments():
    first = json.dumps({"city": "Zürich", "count": 0}, ensure_ascii=False)
    second = '{"enabled":false}'
    calls = collect([
        item_event("added", 1), item_event("added", 3),
        arguments_event("delta", 3, second[:7]),
        arguments_event("done", 1, first),
        arguments_event("delta", 3, second[7:]),
        arguments_event("done", 3, second),
        item_event("done", 1, first), item_event("done", 3, second),
    ])
    assert calls[1] == {"id": "call_1", "name": "operation_1", "arguments": first}
    assert calls[3] == {"id": "call_3", "name": "operation_3", "arguments": second}


@pytest.mark.parametrize("initial", ["", '{"key":"value"}'])
def test_should_preserve_arguments_in_item_events_when_argument_events_are_absent(initial):
    final = '{"key":"value"}'
    assert collect([item_event("added", 0, initial), item_event("done", 0, final)])[0] == {
        "id": "call_0", "name": "operation_0", "arguments": final
    }


def test_should_reject_conflicting_completion_instead_of_corrupting_arguments():
    with pytest.raises(ValueError, match="arguments.*prefix"):
        collect([
            item_event("added", 0),
            arguments_event("delta", 0, '{"x":1'),
            arguments_event("done", 0, '{"x":2}'),
        ])


def test_should_keep_calls_scoped_to_each_stream():
    first = [item_event("added", 0), arguments_event("done", 0, '{"n":1}')]
    second = [item_event("added", 0), arguments_event("done", 0, '{"n":2}')]
    assert collect(first)[0]["arguments"] == '{"n":1}'
    assert collect(second)[0]["arguments"] == '{"n":2}'
