"""Regression test for the Module 10 Faithfulness root-cause fix to
Settings.retrieval_top_k (raised from 5 to 8, see app/core/config.py's
own comment). Confirms the default is actually 8 -- a silent revert of
this config value would reopen the eval-potato-01 regression
(app/core/config.py::retrieval_top_k's docstring has the full
investigation)."""

from app.core.config import settings


def test_retrieval_top_k_default_is_at_least_eight():
    assert settings.retrieval_top_k >= 8, (
        "retrieval_top_k was lowered below the Module 10 Faithfulness fix's minimum -- "
        "see app/core/config.py's comment on this field for why 5 was too narrow "
        "(a genuinely relevant chunk ranked #8 in the hybrid-search candidate pool "
        "for a real benchmark case, eval-potato-01)."
    )
