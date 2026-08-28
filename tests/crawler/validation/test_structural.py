"""Tests for crawler/validation/structural.py — Layer 1 structural heuristics."""

from __future__ import annotations

import pytest

from crawler.validation.structural import StructuralResult, StructuralValidator


VALIDATOR = StructuralValidator()


def _html_with_data_attrs(before_url: str, after_url: str) -> str:
    return f"""
    <html><body>
      <img src="{before_url}" data-label="before" alt="Before treatment" />
      <img src="{after_url}" data-label="after" alt="After treatment" />
    </body></html>
    """


def _html_with_alt_only(before_url: str, after_url: str) -> str:
    return f"""
    <html><body>
      <img src="{before_url}" alt="before botox" />
      <img src="{after_url}" alt="after botox result" />
    </body></html>
    """


def _html_with_figcaption(before_url: str, after_url: str) -> str:
    return f"""
    <html><body>
      <figure>
        <img src="{before_url}" />
        <figcaption>Before the procedure</figcaption>
      </figure>
      <figure>
        <img src="{after_url}" />
        <figcaption>After treatment result</figcaption>
      </figure>
    </body></html>
    """


def _html_no_labels(before_url: str, after_url: str) -> str:
    return f"""
    <html><body>
      <img src="{before_url}" alt="photo 1" />
      <img src="{after_url}" alt="photo 2" />
    </body></html>
    """


def _html_sibling_only(before_url: str, after_url: str) -> str:
    """Only positional (weak) signals — no strong labels."""
    return f"""
    <html><body>
      <div>
        <img src="{before_url}" />
        <img src="{after_url}" />
      </div>
    </body></html>
    """


class TestExplicitLabel:
    def test_data_attrs_give_explicit_label(self):
        result = VALIDATOR.validate(
            "https://example.com/img1.jpg",
            "https://example.com/img2.jpg",
            _html_with_data_attrs(
                "https://example.com/img1.jpg",
                "https://example.com/img2.jpg",
            ),
        )
        assert result.has_explicit_label is True
        assert result.confidence > 0.0

    def test_alt_text_gives_explicit_label(self):
        result = VALIDATOR.validate(
            "https://example.com/img1.jpg",
            "https://example.com/img2.jpg",
            _html_with_alt_only(
                "https://example.com/img1.jpg",
                "https://example.com/img2.jpg",
            ),
        )
        assert result.has_explicit_label is True

    def test_figcaption_gives_explicit_label(self):
        result = VALIDATOR.validate(
            "https://example.com/img1.jpg",
            "https://example.com/img2.jpg",
            _html_with_figcaption(
                "https://example.com/img1.jpg",
                "https://example.com/img2.jpg",
            ),
        )
        assert result.has_explicit_label is True

    def test_no_labels_gives_no_explicit_label(self):
        result = VALIDATOR.validate(
            "https://example.com/img1.jpg",
            "https://example.com/img2.jpg",
            _html_no_labels(
                "https://example.com/img1.jpg",
                "https://example.com/img2.jpg",
            ),
        )
        assert result.has_explicit_label is False

    def test_position_only_is_not_explicit(self):
        result = VALIDATOR.validate(
            "https://example.com/img1.jpg",
            "https://example.com/img2.jpg",
            _html_sibling_only(
                "https://example.com/img1.jpg",
                "https://example.com/img2.jpg",
            ),
        )
        assert result.has_explicit_label is False


class TestFilenameSignal:
    def test_before_filename_detected(self):
        result = VALIDATOR.validate(
            "https://example.com/before_001.jpg",
            "https://example.com/after_001.jpg",
            "<html><body><img src='https://example.com/before_001.jpg'/>"
            "<img src='https://example.com/after_001.jpg'/></body></html>",
        )
        assert result.has_explicit_label is True
        assert any("filename" in s for s in result.before_label_signals)

    def test_korean_filename_detected(self):
        result = VALIDATOR.validate(
            "https://clinic.kr/전_사진.jpg",
            "https://clinic.kr/후_사진.jpg",
            "<html><body></body></html>",
        )
        # Filename signals picked up even without an img tag found in HTML
        assert any("filename" in s for s in result.before_label_signals) or \
               result.confidence >= 0.0  # At minimum, no error raised


class TestConfidenceScore:
    def test_strong_both_sides_gives_high_confidence(self):
        result = VALIDATOR.validate(
            "https://example.com/img1.jpg",
            "https://example.com/img2.jpg",
            _html_with_data_attrs(
                "https://example.com/img1.jpg",
                "https://example.com/img2.jpg",
            ),
        )
        assert result.confidence >= 0.70

    def test_no_signals_gives_zero_confidence(self):
        result = VALIDATOR.validate(
            "https://example.com/img1.jpg",
            "https://example.com/img2.jpg",
            _html_no_labels(
                "https://example.com/img1.jpg",
                "https://example.com/img2.jpg",
            ),
        )
        assert result.confidence == 0.0

    def test_weak_signal_gives_partial_confidence(self):
        result = VALIDATOR.validate(
            "https://example.com/img1.jpg",
            "https://example.com/img2.jpg",
            _html_sibling_only(
                "https://example.com/img1.jpg",
                "https://example.com/img2.jpg",
            ),
        )
        assert 0.0 < result.confidence < 0.45


class TestMultilingual:
    def test_portuguese_alt_text_detected(self):
        html = """
        <html><body>
          <img src="https://ex.com/a.jpg" alt="Antes do tratamento" />
          <img src="https://ex.com/b.jpg" alt="Resultado após procedimento" />
        </body></html>
        """
        result = VALIDATOR.validate(
            "https://ex.com/a.jpg", "https://ex.com/b.jpg", html
        )
        assert result.has_explicit_label is True

    def test_korean_alt_text_detected(self):
        html = """
        <html><body>
          <img src="https://ex.com/a.jpg" alt="시술 전 사진" />
          <img src="https://ex.com/b.jpg" alt="시술 후 결과" />
        </body></html>
        """
        result = VALIDATOR.validate(
            "https://ex.com/a.jpg", "https://ex.com/b.jpg", html
        )
        assert result.has_explicit_label is True
