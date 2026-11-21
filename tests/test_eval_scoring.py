"""Focused tests for evaluation scoring helpers."""

from eval.scoring import combined_score, keyword_score, safety_score, sql_number_match


def test_keyword_score_reports_rate_and_cited_source_hit() -> None:
    score = keyword_score(
        "A water leak was reported.",
        [{"snippet": "Owner reported water entering the cabin."}],
        ["water", "cabin", "airbag"],
    )

    assert score["keyword_hit_rate"] == 1 / 3
    assert score["source_keyword_hit"] is True


def test_empty_keyword_list_is_unscored() -> None:
    assert keyword_score("any answer", [], []) == {
        "status": "unscored",
        "keyword_hit_rate": None,
        "source_keyword_hit": None,
    }


def test_sql_number_match_handles_grouped_and_decimal_values() -> None:
    score = sql_number_match(
        "Door seal: 3.206 (457 defects); other parts: 2.697 (1,332 defects).",
        [{"door_avg": 3.206, "door_count": 457, "other_avg": 2.697, "other_count": 1332}],
    )

    assert score["number_match"] is True
    assert score["missing_numbers"] == []


def test_sql_number_match_fails_when_expected_value_is_missing() -> None:
    assert sql_number_match("133 defects", [{"count": 132}])["number_match"] is False


def test_safety_and_combined_scoring() -> None:
    assert safety_score("refuse", {"refused": True})["behavior_match"] is True
    assert safety_score("not_found", {"answer": "No rows found", "rows": []})["behavior_match"] is True
    assert combined_score(
        {"supporting_data": [{"rows": [{"count": 3}]}], "supporting_documents": [{"id": "complaint:1"}]}
    )["combined_match"] is True
