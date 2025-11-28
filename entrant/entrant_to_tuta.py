from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


def entrant_cell_to_tuta(cell_ent: Dict[str, Any]) -> Dict[str, Any]:
    def int_to_argb_hex(x: int, default: str) -> str:
        if not x:
            return default
        return f"#{x:08x}"

    cell_tuta: Dict[str, Any] = {}

    cell_tuta["T"] = cell_ent.get("T", cell_ent.get("value", ""))
    cell_tuta["V"] = cell_ent.get("V", cell_ent.get("value", ""))
    cell_tuta["NS"] = cell_ent.get("NS", "@")
    cell_tuta["DT"] = cell_ent.get("DT", 0)

    # header flag
    cell_tuta["HF"] = 1 if cell_ent.get("is_header") else 0

    cell_tuta["A1"] = cell_ent.get("A1", "")
    cell_tuta["R1"] = cell_ent.get("R1", "")
    cell_tuta["O"] = cell_ent.get("O", None)

    cell_tuta["LB"] = cell_ent.get("LB", 0)
    cell_tuta["TB"] = cell_ent.get("TB", 0)
    cell_tuta["BB"] = cell_ent.get("BB", 0)
    cell_tuta["RB"] = cell_ent.get("RB", 0)

    bc_ent = cell_ent.get("BC", 0)
    fc_ent = cell_ent.get("FC", 0)
    cell_tuta["BC"] = int_to_argb_hex(bc_ent, "#00ffffff")
    cell_tuta["FC"] = int_to_argb_hex(fc_ent, "#ff000000")

    cell_tuta["FB"] = cell_ent.get("FB", 0)
    cell_tuta["I"] = cell_ent.get("I", 0)
    cell_tuta["HA"] = cell_ent.get("HA", 0)
    cell_tuta["VA"] = cell_ent.get("VA", 2)

    return cell_tuta


def build_2d_grid_from_coordinates(entrant_cells: Sequence[Dict[str, Any]]) -> List[List[Any]]:
    if not entrant_cells:
        return []

    max_row = max(cell["coordinates"][0] for cell in entrant_cells)
    max_col = max(cell["coordinates"][1] for cell in entrant_cells)

    grid: List[List[Any]] = [
        [None for _ in range(max_col + 1)]
        for _ in range(max_row + 1)
    ]

    for cell_ent in entrant_cells:
        r, c = cell_ent["coordinates"]
        grid[r][c] = entrant_cell_to_tuta(cell_ent)

    return grid


def convert_cells(cells: Any) -> List[List[Any]]:
    if not cells:
        return []

    # If already 2D grid
    if isinstance(cells, list) and cells and isinstance(cells[0], list):
        return [
            [
                entrant_cell_to_tuta(cell) if isinstance(cell, dict) else cell
                for cell in row
            ]
            for row in cells
        ]

    # Otherwise expect flat list with coordinates
    if isinstance(cells, list):
        return build_2d_grid_from_coordinates(cells)

    raise TypeError(f"Unsupported cells format: {type(cells).__name__}")


def _capitalized_key(key: str) -> str:
    return key[:1].upper() + key[1:] if key else key


def convert_table(entrant_table: Dict[str, Any], *, table_index: int, src_path: Path) -> Dict[str, Any]:
    cells = entrant_table.get("Cells")
    if cells is None:
        cells = entrant_table.get("cells")
    if cells is None:
        raise KeyError(f"Missing 'Cells' (or 'cells') in {src_path.name} table #{table_index}")

    tuta_grid = convert_cells(cells)

    converted = {
        _capitalized_key(k): v
        for k, v in entrant_table.items()
        if k not in ("Cells", "cells")
    }
    converted["Cells"] = tuta_grid
    return converted


def convert_file(src_path: Path, dst_path: Path) -> None:
    with src_path.open("r", encoding="utf-8") as f:
        entrant_data = json.load(f)

    if not isinstance(entrant_data, list):
        raise TypeError(f"Expected a list at root of {src_path.name}, got {type(entrant_data).__name__}")

    tuta_structure = [
        convert_table(table, table_index=i, src_path=src_path)
        for i, table in enumerate(entrant_data)
    ]

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(tuta_structure, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "entrant_output"
    dst_dir = repo_root / "entrant_to_tuta"

    json_files = sorted(src_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {src_dir}")

    for src_path in json_files:
        dst_path = dst_dir / src_path.name
        convert_file(src_path, dst_path)
        print(f"Converted {src_path.relative_to(repo_root)} -> {dst_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
