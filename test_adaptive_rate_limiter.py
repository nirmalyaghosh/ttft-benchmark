"""
Tests for AdaptiveRateLimiter.

Validates interval adjustment logic, boundary
enforcement, retry behavior, and edge cases.
No external API calls required.
"""

import pytest

from unittest.mock import (
    MagicMock,
    patch,
)

from openai import RateLimitError

from ttft_benchmark import AdaptiveRateLimiter


class TestIntervalAdjustment:
    """
    Interval increases on rate limit,
    decreases on success.
    """

    def test_record_success_decreases_interval(
        self,
    ) -> None:
        """
        Interval decreases by decrease_factor
        after a successful request.
        """
        limiter = AdaptiveRateLimiter(
            min_interval=1.0,
            max_interval=30.0,
            decrease_factor=0.9,
        )
        # Push interval above min so decrease
        # is observable
        limiter.record_rate_limit()
        elevated = limiter.current_interval
        limiter.record_success()
        assert limiter.current_interval < elevated
        assert limiter.current_interval == (
            elevated * 0.9
        )

    def test_record_rate_limit_increases_interval(
        self,
    ) -> None:
        """
        Interval increases by increase_factor
        after a 429 response.
        """
        limiter = AdaptiveRateLimiter(
            min_interval=2.0,
            max_interval=30.0,
            increase_factor=2.0,
        )
        initial = limiter.current_interval
        limiter.record_rate_limit()
        assert limiter.current_interval > initial
        assert limiter.current_interval == (
            initial * 2.0
        )

    def test_rate_limit_increments_counter(
        self,
    ) -> None:
        """
        Each rate limit increments the
        total_rate_limits counter.
        """
        limiter = AdaptiveRateLimiter()
        assert limiter.total_rate_limits == 0
        limiter.record_rate_limit()
        limiter.record_rate_limit()
        assert limiter.total_rate_limits == 2


class TestIntervalBounds:
    """
    Interval stays within min/max bounds.
    """

    def test_interval_does_not_go_below_min(
        self,
    ) -> None:
        """
        Repeated successes cannot push interval
        below min_interval.
        """
        limiter = AdaptiveRateLimiter(
            min_interval=1.0,
            max_interval=30.0,
            decrease_factor=0.5,
        )
        for _ in range(50):
            limiter.record_success()
        assert limiter.current_interval >= 1.0

    def test_interval_does_not_exceed_max(
        self,
    ) -> None:
        """
        Repeated rate limits cannot push
        interval above max_interval.
        """
        limiter = AdaptiveRateLimiter(
            min_interval=1.0,
            max_interval=10.0,
            increase_factor=3.0,
        )
        for _ in range(20):
            limiter.record_rate_limit()
        assert limiter.current_interval <= 10.0


class TestExecuteWithRetry:
    """
    Retry logic for rate-limited API calls.
    """

    @patch("time.sleep")
    def test_returns_result_on_success(
        self,
        mock_sleep: MagicMock,
    ) -> None:
        """
        Successful call returns the function's
        result without retrying.
        """
        limiter = AdaptiveRateLimiter()
        func = MagicMock(return_value="ok")
        result = limiter.execute_with_retry(func)
        assert result == "ok"
        func.assert_called_once()

    @patch("time.sleep")
    def test_returns_none_when_func_returns_none(
        self,
        mock_sleep: MagicMock,
    ) -> None:
        """
        Function returning None is a valid
        result, not a fall-through.
        """
        limiter = AdaptiveRateLimiter()
        func = MagicMock(return_value=None)
        result = limiter.execute_with_retry(func)
        assert result is None
        func.assert_called_once()

    @patch("time.sleep")
    def test_retries_on_rate_limit_then_succeeds(
        self,
        mock_sleep: MagicMock,
    ) -> None:
        """
        Retries after RateLimitError and
        returns result on subsequent success.
        """
        limiter = AdaptiveRateLimiter(
            max_retries=3,
        )
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        error = RateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body=None,
        )
        func = MagicMock(
            side_effect=[error, "ok"],
        )
        result = limiter.execute_with_retry(func)
        assert result == "ok"
        assert func.call_count == 2
        assert limiter.total_retries == 1

    @patch("time.sleep")
    def test_raises_after_max_retries_exhausted(
        self,
        mock_sleep: MagicMock,
    ) -> None:
        """
        Raises RateLimitError after all retries
        are exhausted.
        """
        limiter = AdaptiveRateLimiter(
            max_retries=2,
        )
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        error = RateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body=None,
        )
        func = MagicMock(side_effect=error)
        with pytest.raises(RateLimitError):
            limiter.execute_with_retry(func)
        assert func.call_count == 2
        assert limiter.total_retries == 2

    @patch("time.sleep")
    def test_passes_kwargs_to_func(
        self,
        mock_sleep: MagicMock,
    ) -> None:
        """
        Keyword arguments are forwarded to
        the wrapped function.
        """
        limiter = AdaptiveRateLimiter()
        func = MagicMock(return_value="ok")
        limiter.execute_with_retry(
            func, run_id="abc", prompt="test",
        )
        func.assert_called_once_with(
            run_id="abc", prompt="test",
        )


class TestSummary:
    """
    Summary string for reporting.
    """

    def test_summary_contains_counts(
        self,
    ) -> None:
        """
        Summary includes rate limit and retry
        counts.
        """
        limiter = AdaptiveRateLimiter()
        limiter.record_rate_limit()
        limiter.total_retries = 3
        summary = limiter.summary()
        assert "Rate limits hit: 1" in summary
        assert "Total retries: 3" in summary
