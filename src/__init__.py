# This file makes the src directory a Python package

# Import key classes to make them available at the package level
from .data_loader import DataLoader, DataSource
from .query_processor import QueryProcessor, QueryResult

__all__ = ['DataLoader', 'DataSource', 'QueryProcessor', 'QueryResult']
