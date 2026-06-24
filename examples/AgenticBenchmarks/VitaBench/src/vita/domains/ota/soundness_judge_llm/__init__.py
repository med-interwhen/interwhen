"""
LLM-based soundness judge for OTA.

Uses an LLM to evaluate whether a write tool call (create/pay/cancel/modify) is
consistent with the user's instructions, given the tool call history so far.

Usage:
    from vita.domains.ota.soundness_judge_llm import SoundnessJudge, SoundnessJudgeConfig
"""

from vita.domains.ota.soundness_judge_llm.judge import SoundnessJudge, SoundnessJudgeConfig

__all__ = ["SoundnessJudge", "SoundnessJudgeConfig"]
