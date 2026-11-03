import pytest

from hungercue.timeutil import now_local, parse_hhmm


def test_parse_hhmm() -> None:
    assert parse_hhmm("09:30") == (9, 30)
    for bad in ["25:00", "9", "ab:cd"]:
        with pytest.raises(ValueError):
            parse_hhmm(bad)


def test_now_local_is_naive() -> None:
    assert now_local().tzinfo is None
