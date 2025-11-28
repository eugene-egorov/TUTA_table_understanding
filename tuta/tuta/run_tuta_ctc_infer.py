#!/usr/bin/env python3
"""
Run a fine-tuned TUTA cell-type classifier on prepared `.pt` tables.

The script expects `.pt` files produced by `prepare.py` (pickle dumps of
(token_matrix, number_matrix, position_lists, header_info, format_matrix)).
It builds minimal CTC inputs, runs the model, and writes per-table JSON
with predictions for every non-empty cell.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

# Make local imports work when running from the repo root.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import model.backbones as bbs  # noqa: E402
import model.heads as hds  # noqa: E402
import tokenizer as tknr  # noqa: E402
import utils as ut  # noqa: E402


class CtcInferenceModel(torch.nn.Module):
    """Lightweight wrapper to pair a backbone with the CTC head."""

    def __init__(self, config: argparse.Namespace):
        super().__init__()
        self.target = config.target
        self.backbone = bbs.BACKBONES[config.target](config)
        self.ctc_head = hds.CtcHead(config)

    def forward(
        self,
        token_id: torch.Tensor,
        num_mag: torch.Tensor,
        num_pre: torch.Tensor,
        num_top: torch.Tensor,
        num_low: torch.Tensor,
        token_order: torch.Tensor,
        pos_row: torch.Tensor,
        pos_col: torch.Tensor,
        pos_top: torch.Tensor,
        pos_left: torch.Tensor,
        format_vec: torch.Tensor,
        indicator: torch.Tensor,
        ctc_label: torch.Tensor,
    ):
        if self.target == "base":
            encoded_states = self.backbone(
                token_id,
                num_mag,
                num_pre,
                num_top,
                num_low,
                token_order,
                pos_top,
                pos_left,
                format_vec,
                indicator,
            )
        else:
            encoded_states = self.backbone(
                token_id,
                num_mag,
                num_pre,
                num_top,
                num_low,
                token_order,
                pos_row,
                pos_col,
                pos_top,
                pos_left,
                format_vec,
                indicator,
            )
        return self.ctc_head(encoded_states, indicator, ctc_label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run TUTA CTC inference over prepared `.pt` tables.",
    )

    # Required paths
    parser.add_argument("--model_path", required=True, help="Path to the fine-tuned CTC checkpoint.")
    parser.add_argument("--data_dir", required=True, help="Directory with `.pt` files from prepare.py.")
    parser.add_argument("--output_dir", required=True, help="Where to write JSON prediction files.")

    # Tokenizer / vocab
    parser.add_argument("--vocab_path", type=str, default=str(SCRIPT_DIR / "vocab/bert_vocab.txt"))
    parser.add_argument("--context_repo_path", type=str, default=str(SCRIPT_DIR / "vocab/context_repo_init.txt"))
    parser.add_argument("--cellstr_repo_path", type=str, default=str(SCRIPT_DIR / "vocab/cellstr_repo_init.txt"))

    # Model hyper-params (must match the checkpoint)
    parser.add_argument("--target", type=str, default="tuta", choices=["tuta", "tuta_explicit", "base"])
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--magnitude_size", type=int, default=10)
    parser.add_argument("--precision_size", type=int, default=10)
    parser.add_argument("--top_digit_size", type=int, default=10)
    parser.add_argument("--low_digit_size", type=int, default=10)
    parser.add_argument("--row_size", type=int, default=256)
    parser.add_argument("--column_size", type=int, default=256)
    parser.add_argument("--tree_depth", type=int, default=4)
    parser.add_argument("--node_degree", type=str, default="32,32,64,256")
    parser.add_argument("--num_format_feature", type=int, default=11)
    parser.add_argument("--attention_distance", type=int, default=2)
    parser.add_argument("--attention_step", type=int, default=0)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--num_encoder_layers", type=int, default=12)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--attention_dropout_prob", type=float, default=0.1)
    parser.add_argument("--layer_norm_eps", type=float, default=1e-6)
    parser.add_argument("--hidden_act", type=str, default="gelu")
    parser.add_argument("--attn_method", type=str, default="add", choices=["add", "max"])
    parser.add_argument("--aggregator", type=str, default="sum", choices=["sum", "avg"])
    parser.add_argument("--num_ctc_type", type=int, default=6)

    # Data shaping
    parser.add_argument("--max_seq_len", type=int, default=512, help="Skip tables that exceed this length.")
    parser.add_argument("--max_cell_length", type=int, default=64)
    parser.add_argument("--text_threshold", type=float, default=0.5)
    parser.add_argument("--value_threshold", type=float, default=0.1)
    parser.add_argument("--clc_rate", type=float, default=0.3)
    parser.add_argument("--wcm_rate", type=float, default=0.3)
    parser.add_argument("--add_separate", dest="add_separate", action="store_true", default=True)
    parser.add_argument(
        "--no-add-separate",
        dest="add_separate",
        action="store_false",
        help="Disable adding [SEP] during sequence build (use only if your tokens lack leading [SEP]).",
    )
    parser.add_argument("--sep_or_tok", type=int, default=0, choices=[0, 1], help="Where to drop supervision during sequence build.")

    # Inference options
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--primary_head", type=str, default="tok", choices=["tok", "sep"], help="Which head to expose as `label`.")
    parser.add_argument(
        "--label_map",
        type=str,
        default=None,
        help="Optional comma-separated label names or path to JSON list/dict to map ids to names.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print extra progress info.")

    args = parser.parse_args()
    args.node_degree = [int(deg) for deg in args.node_degree.split(",")]
    args.total_node = sum(args.node_degree)
    return args


def load_label_map(raw: Optional[str], num_classes: int) -> Optional[Dict[int, str]]:
    if raw is None:
        return None
    path = Path(raw)
    names: Sequence[str]
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {int(k): str(v) for k, v in data.items()}
        if isinstance(data, list):
            names = [str(v) for v in data]
        else:
            raise ValueError(f"Unsupported label_map content in {raw}")
    else:
        names = [name.strip() for name in raw.split(",") if name.strip()]
    if len(names) != num_classes:
        print(f"[WARN] label_map size {len(names)} != num_ctc_type {num_classes}; ids will be left as-is.")
        return None
    return {i: name for i, name in enumerate(names)}


def iter_pt_instances(pt_path: Path) -> Iterable[tuple]:
    """Yield every table instance stored in a prepared `.pt` file."""
    with pt_path.open("rb") as f:
        while True:
            try:
                chunk = pickle.load(f)
            except EOFError:
                break
            if isinstance(chunk, list):
                for item in chunk:
                    yield item
            else:
                yield chunk


def build_features_from_instance(
    instance: tuple, tokenizer: tknr.CtcTokenizer, args: argparse.Namespace
) -> Optional[Tuple[Tuple[List, ...], List[Tuple[int, int]]]]:
    """Convert one prepared table into CTC model inputs and cell positions."""
    try:
        token_matrix, number_matrix, position_lists, header_info, fmt_or_text = instance
    except ValueError:
        print("[WARN] unexpected instance structure, skipping.")
        return None

    if not isinstance(token_matrix, list) or not token_matrix:
        return None

    if isinstance(fmt_or_text, list):
        format_matrix = fmt_or_text
    else:
        format_matrix = [
            [tokenizer.default_format for _ in token_matrix[0]] for _ in token_matrix
        ]

    # Use a dummy label matrix that keeps all non-empty cells.
    label_matrix = [[2 for _ in row] for row in token_matrix]
    sampling_matrix = tokenizer.sampling(token_matrix, number_matrix, header_info, label_matrix)
    seq = tokenizer.create_table_seq(
        sampling_matrix=sampling_matrix,
        token_matrix=token_matrix,
        number_matrix=number_matrix,
        position_lists=position_lists,
        format_matrix=format_matrix,
        label_matrix=label_matrix,
        sep_or_tok=args.sep_or_tok,
        add_sep=args.add_separate,
    )
    if seq is None:
        return None
    token_list, num_list, pos_list, format_list, ind_list, label_list = seq
    return flatten_to_features(
        (token_list, num_list, pos_list, format_list, ind_list, label_list), args
    )


def flatten_to_features(
    lists: Tuple[List, List, List, List, List, List], args: argparse.Namespace
) -> Optional[Tuple[Tuple[List, ...], List[Tuple[int, int]]]]:
    token_list, num_list, pos_list, format_list, ind_list, label_list = lists

    token_id: List[int] = []
    num_mag: List[int] = []
    num_pre: List[int] = []
    num_top: List[int] = []
    num_low: List[int] = []
    token_order: List[int] = []
    pos_row: List[int] = []
    pos_col: List[int] = []
    pos_top: List[List[int]] = []
    pos_left: List[List[int]] = []
    format_vec: List[List[float]] = []
    indicator: List[int] = []
    ctc_label: List[int] = []

    cell_positions: List[Tuple[int, int]] = []

    for idx, (tokens, num_feats, (row, col, ttop, tleft), fmt, ind, lbl) in enumerate(
        zip(token_list, num_list, pos_list, format_list, ind_list, label_list)
    ):
        cell_len = len(tokens)
        token_id.extend(tokens)
        num_mag.extend([f[0] for f in num_feats])
        num_pre.extend([f[1] for f in num_feats])
        num_top.extend([f[2] for f in num_feats])
        num_low.extend([f[3] for f in num_feats])

        token_order.extend(list(range(cell_len)))
        pos_row.extend([row for _ in range(cell_len)])
        pos_col.extend([col for _ in range(cell_len)])
        entire_top = ut.UNZIPS[args.target](ttop, args.node_degree, args.total_node)
        entire_left = ut.UNZIPS[args.target](tleft, args.node_degree, args.total_node)
        pos_top.extend([entire_top for _ in range(cell_len)])
        pos_left.extend([entire_left for _ in range(cell_len)])

        format_vec.extend([fmt for _ in range(cell_len)])
        indicator.extend(ind)
        ctc_label.extend(lbl)

        # idx==0 corresponds to the leading [CLS]
        if idx > 0:
            cell_positions.append((int(row), int(col)))

    if args.max_seq_len and len(token_id) > args.max_seq_len:
        return None

    features = (
        token_id,
        num_mag,
        num_pre,
        num_top,
        num_low,
        token_order,
        pos_row,
        pos_col,
        pos_top,
        pos_left,
        format_vec,
        indicator,
        ctc_label,
    )
    return features, cell_positions


def build_model(
    args: argparse.Namespace, device: torch.device
) -> Tuple[CtcInferenceModel, tknr.CtcTokenizer]:
    tokenizer = tknr.CtcTokenizer(args)
    args.vocab_size = len(tokenizer.vocab)

    model = CtcInferenceModel(args)
    state = torch.load(args.model_path, map_location=device)

    if isinstance(state, torch.nn.Module):
        model = state
    elif isinstance(state, dict):
        state_dict = state.get("model_state_dict", state)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if args.verbose:
            print(f"[INFO] Loaded state_dict with {len(missing)} missing and {len(unexpected)} unexpected keys.")
    else:
        raise RuntimeError(f"Unsupported checkpoint format at {args.model_path}")

    model.to(device)
    model.eval()
    return model, tokenizer


def id_to_label(idx: int, label_map: Optional[Dict[int, str]]) -> Union[str, int]:
    if label_map is None:
        return int(idx)
    return label_map.get(int(idx), str(idx))


def predict_single(
    model: CtcInferenceModel, features: Tuple[List, ...], device: torch.device
) -> Tuple[List[int], List[int]]:
    (
        token_id,
        num_mag,
        num_pre,
        num_top,
        num_low,
        token_order,
        pos_row,
        pos_col,
        pos_top,
        pos_left,
        format_vec,
        indicator,
        ctc_label,
    ) = features

    def to_long(data: List) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.long, device=device).unsqueeze(0)

    inputs = {
        "token_id": to_long(token_id),
        "num_mag": to_long(num_mag),
        "num_pre": to_long(num_pre),
        "num_top": to_long(num_top),
        "num_low": to_long(num_low),
        "token_order": to_long(token_order),
        "pos_row": to_long(pos_row),
        "pos_col": to_long(pos_col),
        "pos_top": torch.tensor(pos_top, dtype=torch.long, device=device).unsqueeze(0),
        "pos_left": torch.tensor(pos_left, dtype=torch.long, device=device).unsqueeze(0),
        "format_vec": torch.tensor(format_vec, dtype=torch.float, device=device).unsqueeze(0),
        "indicator": to_long(indicator),
        "ctc_label": to_long(ctc_label),
    }

    with torch.no_grad():
        sep_triple, tok_triple = model(**inputs)

    sep_pred = sep_triple[1].detach().cpu().tolist()
    tok_pred = tok_triple[1].detach().cpu().tolist()
    return sep_pred, tok_pred


def run_inference(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    label_map = load_label_map(args.label_map, args.num_ctc_type)
    model, tokenizer = build_model(args, device)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(data_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {data_dir}")

    for pt_path in pt_files:
        tables_json: List[Dict] = []
        table_idx = 0
        for instance in iter_pt_instances(pt_path):
            built = build_features_from_instance(instance, tokenizer, args)
            if built is None:
                if args.verbose:
                    print(f"[WARN] Skip table #{table_idx} in {pt_path.name} (build failed or too long).")
                table_idx += 1
                continue
            features, cell_positions = built
            sep_pred, tok_pred = predict_single(model, features, device)

            # Align predictions with cell coordinates
            cell_entries = []
            paired_len = min(len(cell_positions), len(tok_pred), len(sep_pred))
            for i in range(paired_len):
                row, col = cell_positions[i]
                sep_label = id_to_label(sep_pred[i], label_map)
                tok_label = id_to_label(tok_pred[i], label_map)
                cell_entry = {
                    "row": row,
                    "col": col,
                    "label_sep": sep_label,
                    "label_tok": tok_label,
                    "label": tok_label if args.primary_head == "tok" else sep_label,
                }
                cell_entries.append(cell_entry)

            tables_json.append(
                {
                    "table_index": table_idx,
                    "source": pt_path.name,
                    "cells": cell_entries,
                }
            )
            table_idx += 1

        out_path = output_dir / f"{pt_path.stem}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(tables_json, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Wrote {len(tables_json)} tables to {out_path}")


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
