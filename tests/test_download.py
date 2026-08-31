from __future__ import annotations

import pytest

from flight_delay_milp.download import validate_bts_url


@pytest.mark.parametrize(
    "url",
    [
        "https://www.transtats.bts.gov/OT_Delay/export.zip",
        "https://transtats.bts.gov/example.zip",
    ],
)
def test_accepts_official_bts_https_url(url: str) -> None:
    validate_bts_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://www.transtats.bts.gov/export.zip",
        "https://transtats.bts.gov.evil.example/export.zip",
        "https://example.com/export.zip",
    ],
)
def test_rejects_non_official_download(url: str) -> None:
    with pytest.raises(ValueError):
        validate_bts_url(url)
