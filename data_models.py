#!/usr/bin/env python3

"""
Defines the core data structures for the application.
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from functools import total_ordering
from typing import Tuple

@total_ordering
@dataclass
class ReceiptItem:
    """Represents a single item parsed from a receipt."""
    pid: int
    status_flag: str
    date: str
    order_number: int
    name: str
    size: float = np.nan
    price: float = np.nan
    price_per_one: float = np.nan
    shop: str = "denner"
    discount: float = 0.0

    def _get_sort_tuple(self) -> Tuple[datetime, int]:
        """Creates a tuple used for sorting items, primarily by date."""
        try:
            date_obj = datetime.strptime(self.date, '%d.%m.%Y')
        except (ValueError, TypeError):
            date_obj = datetime.max  # Sort items with invalid dates to the end
        return (date_obj, self.pid)

    def __lt__(self, other):
        if not isinstance(other, ReceiptItem):
            return NotImplemented
        return self._get_sort_tuple() < other._get_sort_tuple()

    def __eq__(self, other):
        if not isinstance(other, ReceiptItem):
            return NotImplemented
        return self._get_sort_tuple() == other._get_sort_tuple()