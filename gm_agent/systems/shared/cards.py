"""Standard playing card deck simulation.

Used by Delve (card-driven exploration) and potentially
The Quiet Year and other card-driven games.
"""

import random
from dataclasses import dataclass


SUITS = ("hearts", "diamonds", "clubs", "spades")
RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")


@dataclass
class Card:
    """A single playing card."""
    suit: str
    rank: str

    def __str__(self) -> str:
        return f"{self.rank} of {self.suit}"

    @property
    def is_face(self) -> bool:
        return self.rank in ("J", "Q", "K")

    @property
    def numeric_value(self) -> int:
        """Numeric value (A=1, 2-10=face, J=11, Q=12, K=13)."""
        if self.rank == "A":
            return 1
        if self.rank in ("J", "Q", "K"):
            return {"J": 11, "Q": 12, "K": 13}[self.rank]
        return int(self.rank)


class CardDeck:
    """A standard 52-card deck with optional jokers."""

    def __init__(self, include_jokers: bool = False):
        self._cards: list[Card] = []
        self._include_jokers = include_jokers
        self.shuffle()

    def shuffle(self) -> None:
        """Rebuild and shuffle the deck."""
        self._cards = [Card(suit, rank) for suit in SUITS for rank in RANKS]
        if self._include_jokers:
            self._cards.append(Card("joker", "Red"))
            self._cards.append(Card("joker", "Black"))
        random.shuffle(self._cards)

    def draw(self) -> Card | None:
        """Draw a card from the top. Returns None if empty."""
        if not self._cards:
            return None
        return self._cards.pop()

    def remaining(self) -> int:
        """Number of cards remaining in the deck."""
        return len(self._cards)

    def peek(self) -> Card | None:
        """Look at the top card without removing it."""
        if not self._cards:
            return None
        return self._cards[-1]

    def is_empty(self) -> bool:
        """Check if the deck is empty."""
        return len(self._cards) == 0
