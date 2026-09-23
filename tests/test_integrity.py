"""Unit tests for the word error rate calculation used by the content integrity check."""

from app.integrity import word_error_rate


def test_wer_identical_text_is_zero():
    assert word_error_rate("the quick brown fox", "the quick brown fox") == 0.0


def test_wer_completely_different_text_is_one():
    assert word_error_rate("the quick brown fox", "totally unrelated words here") == 1.0


def test_wer_one_substitution():
    # 1 error / 4 reference words
    assert word_error_rate("the quick brown fox", "the slow brown fox") == 0.25


def test_wer_one_insertion():
    # hypothesis has an extra word not in reference: 1 edit / 4 reference words
    assert word_error_rate("the quick brown fox", "the very quick brown fox") == 0.25


def test_wer_one_deletion():
    # hypothesis is missing a word: 1 edit / 4 reference words
    assert word_error_rate("the quick brown fox", "the brown fox") == 0.25


def test_wer_empty_reference_and_hypothesis():
    assert word_error_rate("", "") == 0.0


def test_wer_empty_reference_nonempty_hypothesis():
    assert word_error_rate("", "hello") == 1.0


def test_wer_hallucinated_word_flagged_above_threshold():
    # Simulates a generative model inventing content: several changed words.
    reference = "please call me back as soon as possible"
    hypothesis = "please call the police as soon as possible"
    wer = word_error_rate(reference, hypothesis)
    assert wer > 0.05
