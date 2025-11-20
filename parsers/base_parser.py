#!/usr/bin/env python3

"""
Defines the abstract base class for all receipt parsers.

Each shop-specific parser must inherit from BaseParser and implement
all of its abstract methods. This creates a consistent interface for the
main application to use, regardless of the shop.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Set

# We import our data model so the parser knows what it's creating
from data_models import ReceiptItem

class BaseParser(ABC):
    """Abstract base class for all receipt parsers."""
    
    @abstractmethod
    def parse_date(self, text: str) -> Optional[str]:
        """
        Parses the date from the raw OCR text.

        Args:
            text: The full raw text extracted from the receipt.

        Returns:
            The date string in 'dd.mm.YYYY' format, or None if not found.
        """
        pass

    @abstractmethod
    def parse_items(self, text: str, date_str: str, corrections_map: Dict[str, dict], size_map: Dict[str, float]) -> Tuple[List[ReceiptItem], Set]:
        """
        Parses all items from the raw OCR text.

        Args:
            text: The full raw text extracted from the receipt.
            date_str: The date associated with this receipt.
            corrections_map: A map for correcting product names.
            size_map: A map of standard sizes for products.

        Returns:
            A tuple containing:
            - A list of parsed ReceiptItem objects.
            - A set of raw product names that need a size definition.
        """
        pass