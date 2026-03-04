"""Tests for the shared CardDeck utility."""

from gm_agent.systems.shared.cards import Card, CardDeck, SUITS, RANKS


class TestCard:
    def test_str(self):
        c = Card("hearts", "A")
        assert str(c) == "A of hearts"

    def test_face_cards(self):
        assert Card("spades", "J").is_face
        assert Card("spades", "Q").is_face
        assert Card("spades", "K").is_face
        assert not Card("spades", "10").is_face
        assert not Card("spades", "A").is_face

    def test_numeric_values(self):
        assert Card("hearts", "A").numeric_value == 1
        assert Card("hearts", "5").numeric_value == 5
        assert Card("hearts", "10").numeric_value == 10
        assert Card("hearts", "J").numeric_value == 11
        assert Card("hearts", "Q").numeric_value == 12
        assert Card("hearts", "K").numeric_value == 13


class TestCardDeck:
    def test_standard_deck_size(self):
        deck = CardDeck()
        assert deck.remaining() == 52

    def test_deck_with_jokers(self):
        deck = CardDeck(include_jokers=True)
        assert deck.remaining() == 54

    def test_draw_reduces_count(self):
        deck = CardDeck()
        card = deck.draw()
        assert card is not None
        assert deck.remaining() == 51

    def test_draw_returns_card(self):
        deck = CardDeck()
        card = deck.draw()
        assert isinstance(card, Card)
        assert card.suit in SUITS
        assert card.rank in RANKS

    def test_draw_all_cards(self):
        deck = CardDeck()
        cards = []
        for _ in range(52):
            c = deck.draw()
            assert c is not None
            cards.append(c)
        assert deck.remaining() == 0
        assert deck.is_empty()
        # All 52 unique
        card_strs = [str(c) for c in cards]
        assert len(set(card_strs)) == 52

    def test_draw_from_empty(self):
        deck = CardDeck()
        for _ in range(52):
            deck.draw()
        assert deck.draw() is None

    def test_peek_doesnt_remove(self):
        deck = CardDeck()
        top = deck.peek()
        assert top is not None
        assert deck.remaining() == 52
        drawn = deck.draw()
        assert drawn == top

    def test_peek_empty(self):
        deck = CardDeck()
        for _ in range(52):
            deck.draw()
        assert deck.peek() is None

    def test_shuffle_resets(self):
        deck = CardDeck()
        for _ in range(10):
            deck.draw()
        assert deck.remaining() == 42
        deck.shuffle()
        assert deck.remaining() == 52

    def test_is_empty(self):
        deck = CardDeck()
        assert not deck.is_empty()
        for _ in range(52):
            deck.draw()
        assert deck.is_empty()
