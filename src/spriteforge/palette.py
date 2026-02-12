"""Palette system for symbol-to-RGBA mapping and palette swapping."""

from __future__ import annotations

from spriteforge.models import PaletteColor, PaletteConfig


def build_palette_map(
    palette: PaletteConfig,
) -> dict[str, tuple[int, int, int, int]]:
    """Build a complete symbol-to-RGBA mapping from a palette config.

    Includes the transparent symbol (→ (0,0,0,0)) and outline symbol
    in addition to all named palette colors.

    Args:
        palette: The palette configuration to build from.

    Returns:
        A dict mapping each single-character symbol to its RGBA tuple.
    """
    mapping: dict[str, tuple[int, int, int, int]] = {}

    # Transparent symbol always maps to fully transparent
    mapping[palette.transparent_symbol] = (0, 0, 0, 0)

    # Outline symbol maps to its configured RGBA
    mapping[palette.outline.symbol] = palette.outline.rgba

    # Named palette colors
    for color in palette.colors:
        mapping[color.symbol] = color.rgba

    return mapping


def validate_grid_symbols(grid: list[str], palette: PaletteConfig) -> list[str]:
    """Validate that all characters in a grid are valid palette symbols.

    Args:
        grid: List of strings (typically 64 strings, each 64 characters).
        palette: The palette config defining valid symbols.

    Returns:
        List of unique invalid symbols found (empty if all valid).
    """
    valid_symbols = set(build_palette_map(palette).keys())
    invalid: list[str] = []
    seen: set[str] = set()

    for row in grid:
        for ch in row:
            if ch not in valid_symbols and ch not in seen:
                invalid.append(ch)
                seen.add(ch)

    return invalid


def swap_palette_grid(
    grid: list[str],
    from_palette: PaletteConfig,
    to_palette: PaletteConfig,
) -> list[str]:
    """Swap palette symbols in a grid from one palette to another.

    Maps colors by name (e.g., "Skin" in P1 → "Skin" in P2), replacing
    the source symbol with the target symbol. Transparent and outline
    symbols are preserved unchanged.

    Args:
        grid: The source grid using from_palette symbols.
        from_palette: The source palette.
        to_palette: The target palette.

    Returns:
        A new grid with symbols swapped to the target palette.

    Raises:
        ValueError: If palettes have mismatched color names.
    """
    # Build name → symbol maps for both palettes
    from_names = {c.element for c in from_palette.colors}
    to_names = {c.element for c in to_palette.colors}

    missing = from_names - to_names
    if missing:
        raise ValueError(f"Target palette is missing color names: {sorted(missing)}")

    to_by_name: dict[str, str] = {c.element: c.symbol for c in to_palette.colors}

    # Build character swap table (from_symbol → to_symbol)
    swap_table: dict[str, str] = {}
    for color in from_palette.colors:
        swap_table[color.symbol] = to_by_name[color.element]

    # Transparent and outline are preserved (identity mapping)
    swap_table[from_palette.transparent_symbol] = to_palette.transparent_symbol
    swap_table[from_palette.outline.symbol] = to_palette.outline.symbol

    # Apply swap
    new_grid: list[str] = []
    for row in grid:
        new_row = "".join(swap_table.get(ch, ch) for ch in row)
        new_grid.append(new_row)

    return new_grid
