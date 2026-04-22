"""
vocab_config.py — Tool Vocabulary Size Configuration
======================================================

Controls how many tools are active across the entire codebase.
Change ``ACTIVE_TOOL_COUNT`` to 15, 30, or 45 to switch tiers.

    ACTIVE_TOOL_COUNT = 15   →  original 15-tool baseline
    ACTIVE_TOOL_COUNT = 30   →  Tier 1 + Tier 2
    ACTIVE_TOOL_COUNT = 45   →  full vocabulary (default)
"""

ACTIVE_TOOL_COUNT: int = 45
