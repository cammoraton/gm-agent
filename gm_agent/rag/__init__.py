"""Pathfinder 2e RAG (Retrieval-Augmented Generation) module.

This module provides search capabilities over Pathfinder 2e content
using SQLite FTS5 for fast keyword search and optional semantic search.
"""

from .search import PathfinderSearch

__all__ = ["PathfinderSearch"]
