"""Backward-compatible entrypoint for serving the API."""

from fake_news_pipeline.api import app


__all__ = ["app"]