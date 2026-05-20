from __future__ import annotations

import unittest

from aether.runtime.core.contracts import TurnContext
from aether.runtime.core.turn_metadata import (
    INTERNAL_METADATA_KEYS,
    RUNTIME_REF_KEYS,
    TURN_METADATA_DEFAULTS,
    get_runtime_ref,
    init_turn_retry_counters,
    public_turn_metadata,
    set_runtime_ref,
)
from aether.runtime.session.session_runtime import (
    TURN_KEY_EMPTY_RECOVERY_LAST_STEP,
    TURN_KEY_EMPTY_RESPONSE_RETRIES,
    TURN_KEY_POST_TOOL_EMPTY_RETRIED,
    TURN_KEY_PROVIDER_ERROR_RETRIES,
    TURN_KEY_STREAMED_ASSISTANT_TEXT,
)


class TurnMetadataContractTests(unittest.TestCase):
    def test_init_turn_retry_counters_sets_existing_defaults(self) -> None:
        metadata: dict[str, object] = {"custom": "keep"}

        init_turn_retry_counters(metadata)

        self.assertEqual(metadata["custom"], "keep")
        self.assertEqual(metadata[TURN_KEY_EMPTY_RESPONSE_RETRIES], 0)
        self.assertEqual(metadata[TURN_KEY_PROVIDER_ERROR_RETRIES], 0)
        self.assertEqual(metadata[TURN_KEY_STREAMED_ASSISTANT_TEXT], "")
        self.assertEqual(metadata[TURN_KEY_EMPTY_RECOVERY_LAST_STEP], "")
        self.assertIs(metadata[TURN_KEY_POST_TOOL_EMPTY_RETRIED], False)
        self.assertEqual(set(TURN_METADATA_DEFAULTS).issubset(metadata), True)

    def test_runtime_refs_are_filtered_from_public_turn_snapshot(self) -> None:
        context = TurnContext(
            session_id="s",
            iteration=1,
            metadata={"visible": {"ok": True}, "usage_accumulator": object()},
        )
        marker = object()
        set_runtime_ref(context, "_engine_config", marker)
        set_runtime_ref(context, "_parent_agent", marker)
        set_runtime_ref(context, "_task_store", marker)

        snapshot = public_turn_metadata(context.metadata)

        self.assertEqual(snapshot, {"visible": {"ok": True}})
        self.assertTrue(RUNTIME_REF_KEYS.issubset(INTERNAL_METADATA_KEYS))

    def test_set_runtime_ref_rejects_unknown_keys(self) -> None:
        context = TurnContext(session_id="s", iteration=1)

        with self.assertRaises(KeyError):
            set_runtime_ref(context, "_not_registered", object())

    def test_get_runtime_ref_can_type_check(self) -> None:
        context = TurnContext(session_id="s", iteration=1)
        set_runtime_ref(context, "_engine_config", "config-ish")

        self.assertEqual(
            get_runtime_ref(context, "_engine_config", str),
            "config-ish",
        )
        self.assertIsNone(get_runtime_ref(context, "_engine_config", dict))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
