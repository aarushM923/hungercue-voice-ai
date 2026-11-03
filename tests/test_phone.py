from hungercue.phone import find_phone, normalize_phone


def test_normalize_us_formats() -> None:
    for raw in ["2125550100", "(212) 555-0100", "212.555.0100", "+1 212 555 0100", "1-212-555-0100"]:
        assert normalize_phone(raw) == "+12125550100"


def test_normalize_rejects_garbage_and_keeps_international() -> None:
    assert normalize_phone("12345") == ""
    assert normalize_phone("") == ""
    assert normalize_phone("+44 20 7946 0958") == "+442079460958"


def test_find_phone_in_speech() -> None:
    assert find_phone("you can reach me at 212 555 0100 thanks") == "+12125550100"
    assert find_phone("no number here") == ""
