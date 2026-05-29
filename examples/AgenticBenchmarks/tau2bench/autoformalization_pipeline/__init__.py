"""Spec pipeline.

Three deterministic stages for turning ``policy.md`` + ``tools.py`` into a
Lean-verified policy checker and Python glue:

* ``generate_spec``  — LLM-driven, produces ``PolicyChecker.lean`` and a
                       structured ``manifest.json``.  Loops on ``lake build``.
* ``generate_runner`` — pure-template, renders ``LeanMain.lean`` from
                        the manifest.
* ``generate_glue``   — pure-template, renders ``telecom_glue_spec.py``
                        from the manifest.

Run end-to-end via :mod:`tau2.verifier.spec_pipeline.cli`.
"""
