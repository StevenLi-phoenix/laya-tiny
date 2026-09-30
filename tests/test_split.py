import pytest

from laya_tiny.split import LeakIndex, shingles

pytestmark = pytest.mark.unit


def test_shingles_short_and_long():
    assert shingles("Reset password?") == frozenset(["reset password"])
    assert "how do i" in shingles("How do I reset my password")
    assert shingles("!!!") == frozenset()


def test_leak_index_finds_near_duplicates():
    idx = LeakIndex(["How do I reset my password?", "The app crashes every time I open settings."])
    j, i = idx.max_jaccard("how do i reset my password")
    assert j == 1.0 and i == 0
    j, i = idx.max_jaccard("The app crashes every time I open the settings page")
    assert 0.4 < j < 1.0 and i == 1
    assert idx.max_jaccard("Completely unrelated invoice question about VAT")[0] == 0.0
