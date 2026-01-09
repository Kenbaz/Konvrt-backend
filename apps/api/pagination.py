"""
Custom pagination classes for the API layer.

This module provides pagination classes that return responses in a
standardized format consistent with the API response structure.
"""
from collections import OrderedDict
from typing import Any, Dict, List

from rest_framework.pagination import PageNumberPagination, LimitOffsetPagination
from rest_framework.response import Response


class StandardPageNumberPagination(PageNumberPagination):
    """
    Standard page number pagination with customized response format.
    
    Supports query parameters:
        - page: Page number (1-indexed)
        - page_size: Number of items per page (optional, uses default)
    """
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100

    def get_paginated_response(self, data: List[Any]) -> Response:
        """
        Return a paginated response in standardized format.
        """
        return Response(OrderedDict([
            ("success", True),
            ("data", data),
            ("metadata", OrderedDict({
                ("pagination", OrderedDict([
                    ("page", self.page.number),
                    ("page_size", self.page.paginator.per_page),
                    ("total_count", self.page.paginator.count),
                    ("total_pages", self.page.paginator.num_pages),
                    ("has_next", self.page.has_next()),
                    ("has_previous", self.page.has_previous()),
                ])),
                ("links", OrderedDict([
                    ("next", self.get_next_link()),
                    ("previous", self.get_previous_link()),
                ])),
            })),
        ]))
    

    def get_paginated_response_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return OpenAPI schema for paginated response.
        """
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "example": True},
                "data": schema,
                "metadata": {
                    "type": "object",
                    "properties": {
                        "pagination": {
                            "type": "object",
                            "properties": {
                                "page": {"type": "integer", "example": 1},
                                "page_size": {"type": "integer", "example": 20},
                                "total_count": {"type": "integer", "example": 100},
                                "total_pages": {"type": "integer", "example": 5},
                                "has_next": {"type": "boolean", "example": True},
                                "has_previous": {"type": "boolean", "example": False},
                            },
                        },
                        "links": {
                            "type": "object",
                            "properties": {
                                "next": {"type": "string", "nullable": True},
                                "previous": {"type": "string", "nullable": True},
                            },
                        },
                    },
                },
            },
        }


class StandardLimitOffsetPagination(LimitOffsetPagination):
    """
    Standard limit/offset pagination with customized response format.
    
    Supports query parameters:
        - limit: Maximum number of items to return
        - offset: Number of items to skip
    """
    default_limit = 20
    max_limit = 100
    
    
    def get_paginated_response(self, data: List[Any]) -> Response:
        """
        Return a paginated response in standardized format.
        """
        return Response(OrderedDict([
            ("success", True),
            ("data", data),
            ("metadata", OrderedDict([
                ("pagination", OrderedDict([
                    ("limit", self.limit),
                    ("offset", self.offset),
                    ("total_count", self.count),
                    ("has_next", self.get_next_link() is not None),
                    ("has_previous", self.get_previous_link() is not None),
                ])),
                ("links", OrderedDict([
                    ("next", self.get_next_link()),
                    ("previous", self.get_previous_link()),
                ])),
            ])),
        ]))
    

    def get_paginated_response_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return OpenAPI schema for paginated response.
        """
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "example": True},
                "data": schema,
                "metadata": {
                    "type": "object",
                    "properties": {
                        "pagination": {
                            "type": "object",
                            "properties": {
                                "limit": {"type": "integer", "example": 20},
                                "offset": {"type": "integer", "example": 0},
                                "total_count": {"type": "integer", "example": 100},
                                "has_next": {"type": "boolean", "example": True},
                                "has_previous": {"type": "boolean", "example": False},
                            },
                        },
                        "links": {
                            "type": "object",
                            "properties": {
                                "next": {"type": "string", "nullable": True},
                                "previous": {"type": "string", "nullable": True},
                            },
                        },
                    },
                },
            },
        }


class OperationPagination(StandardLimitOffsetPagination):
    """
    Pagination specifically for operation listings.
    
    Uses limit/offset pagination with a higher default limit
    since operation lists are typically viewed in full.
    """
    default_limit = 50
    max_limit = 100
