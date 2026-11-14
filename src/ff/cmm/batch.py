"""CT channel types of a species list (the directed donor > acceptor pairs the model resolves energies by)."""
from __future__ import annotations

__all__ = ["channel_types", "CHANNEL_TYPES"]

CHANNEL_TYPES = ("Cl>Na", "Na>Cl", "Cl>Cl", "Na>Na")       # the NaCl legacy order (cached labels and runs)


def channel_types(symbols):
    """Directed channel type names 'D>A' for a species list: the legacy NaCl order for {Na, Cl}, every ordered pair
    otherwise."""
    if set(symbols) == {"Na", "Cl"}:
        return CHANNEL_TYPES
    return tuple(f"{a}>{b}" for a in symbols for b in symbols)
