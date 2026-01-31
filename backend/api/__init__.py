"""
API Package.

This package contains the FastAPI application and all related components:
- routes: API endpoint handlers
- schemas: Pydantic models for request/response validation
- services: Business logic services
"""

from backend.api.app import app, create_app

__all__ = ["app", "create_app"]
