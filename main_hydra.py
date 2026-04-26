#!/usr/bin/env python3
"""
HydraBOA.

2. Builds a HydraBOA model (small d_model for speed).
3. Trains for a handful of epochs — verifies loss decreases.
4. Compresses a test segment with the trained model + range coder.
5. Decompresses it.
6. Verifies byte-exact round-trip.

Usage:
    python main_hydra.py [--device cpu|cuda] [--epochs 10] [--K 4]
"""

import argparse
import csv
import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from hydra_model import HydraBOA
from hydra_codec import compress_hydra, decompress_hydra
try:
    from hydra_codec import compress_hydra_gpu, decompress_hydra_gpu
    _HAS_GPU_CODEC = True
except Exception:
    _HAS_GPU_CODEC = False


def make_synthetic_data(size: int = 20_480, seed: int = 42) -> np.ndarray:
    """Repeating period-8 pattern with small additive noise (compressible)."""
    rng = np.random.RandomState(seed)
    base = np.tile(np.arange(8, dtype=np.uint8) * 30, size // 8 + 1)[:size]
    noise = rng.randint(0, 8, size=size, dtype=np.uint8)
    return ((base.astype(np.int16) + noise) % 256).astype(np.uint8)


def train_hydra(model, data: np.ndarray, *,
                seq_len: int, batch_size: int, num_epochs: int,
                device: str, K: int, lr: float = 1e-3,
                precision: str = "fp32", use_compile: bool = False):
    """Minimal training loop — returns final average bpp."""
    block = seq_len * batch_size
    n_blocks = len(data) // block
    if n_blocks == 0:
        raise ValueError(f"Data too small ({len(data)}) for seq_len={seq_len} × batch_size={batch_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    V = model.vocab_size

    if use_compile:
        print("  [INFO] Compiling model with torch.compile (mode='max-autotune')...")
        model = torch.compile(model, mode="max-autotune")

    model.train()
    print(f"\n{'═'*60}")
    print(f"  Training  |  {num_epochs} epochs, seq_len={seq_len}, bs={batch_size}, "
          f"K={K}, lr={lr}, precision={precision}, device={device}")
    print(f"  Batches per epoch: {n_blocks}")
    print(f"{'═'*60}")

    amp_enabled = precision in ("bf16", "fp16") and device == "cuda"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision, torch.float32)

    best_bpp = float("inf")

    for epoch in range(1, num_epochs + 1):
        epoch_loss, n_tok, pos = 0.0, 0, 0
        t0 = time.perf_counter()
        for _ in range(n_blocks):
            chunk = data[pos : pos + block]
            pos += block
            batch = torch.tensor(
                chunk.reshape(batch_size, seq_len), dtype=torch.long, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(batch)
                loss = criterion(logits.reshape(-1, V), batch.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * batch.numel()
            n_tok += batch.numel()

        avg = epoch_loss / n_tok
        bpp = avg / np.log(2)
        dt = time.perf_counter() - t0
        tok_s = n_tok / max(dt, 1e-6)
        mb_s = n_tok / 1e6 / max(dt, 1e-6)
        marker = " *" if bpp < best_bpp else ""
        best_bpp = min(best_bpp, bpp)
        print(f"  Epoch {epoch:2d}  loss={avg:.4f}  bpp={bpp:.3f}  "
              f"ratio~{8/max(bpp,0.01):.2f}x  {mb_s:.1f} MB/s  {dt:.1f}s{marker}")

    return bpp


def main():
    def _upsert_metrics_row(csv_path: Path, row: dict, key_col: str = "model") -> None:
        if key_col not in row or not str(row.get(key_col, "")).strip():
            return

        rows = []
        fields = []
        if csv_path.exists():
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                fields = list(reader.fieldnames or [])
                rows = list(reader)

        preferred = [
            "model", "compression_ratio", "throughput_compress_MBps",
            "throughput_decompress_MBps", "original_size", "compressed_size",
            "time_compress_s", "time_decompress_s", "experiment",
            "checkpoint_path", "updated_at_utc",
        ]
        merged_fields = []
        for c in preferred + fields + list(row.keys()):
            if c not in merged_fields:
                merged_fields.append(c)

        key = str(row[key_col]).strip()
        found = False
        for r in rows:
            if str(r.get(key_col, "")).strip() == key:
                r.update({k: "" if v is None else str(v) for k, v in row.items()})
                found = True
                break
        if not found:
            nr = {c: "" for c in merged_fields}
            nr.update({k: "" if v is None else str(v) for k, v in row.items()})
            rows.append(nr)

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=merged_fields)
            writer.writeheader()
            for r in rows:
                writer.writerow({c: r.get(c, "") for c in merged_fields})

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--test-bytes", type=int, default=256,
                    help="Bytes to compress/decompress for round-trip verify. "
                         "Use 0 to compress the full training data.")
    ap.add_argument("--data-path", type=str, default=None,
                    help="Path to a real binary dataset file (overrides synthetic).")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--precision", type=str, default="fp32",
                    choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model for faster training.")
    ap.add_argument("--data-frac", type=float, default=1.0,
                    help="Fraction of data to use for training (0.0, 1.0].")
    ap.add_argument("--gpu-codec", action="store_true",
                    help="Use GPU range coder for compress/decompress.")
    ap.add_argument("--save-checkpoint", type=str, default=None,
                    help="Path to save model checkpoint (.pt) after training.")
    ap.add_argument("--load-checkpoint", type=str, default=None,
                    help="Path to load model checkpoint (.pt) and skip training.")
    ap.add_argument("--train-only", action="store_true",
                    help="Only train the model and save checkpoint, then exit.")
    ap.add_argument("--metrics-csv", type=str, default=None,
                    help="Optional CSV path to upsert compare metrics row.")
    ap.add_argument("--metrics-model-name", type=str, default=None,
                    help="Optional model name key for metrics CSV row.")
    args = ap.parse_args()

    K = args.K
    device = args.device
    d_model = args.d_model

    print("╔══════════════════════════════════════════════════════════╗")
    print("║              HydraBOA  —  Smoke Test                    ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Device     : {device}")
    print(f"  K          : {K}")
    print(f"  d_model    : {d_model}")
    print(f"  num_layers : {args.num_layers}")
    print(f"  precision  : {args.precision}")

    # ── 1. Load data ──
    full_raw = None  # keep full file for compress-all
    if args.data_path:
        full_raw = np.fromfile(args.data_path, dtype=np.uint8)
        raw = full_raw.copy()
        # Apply data fraction for training only
        if args.data_frac < 1.0:
            raw = raw[:int(len(raw) * args.data_frac)]
        # Trim to multiple of (seq_len * batch_size) for clean batching
        block = args.seq_len * args.batch_size
        usable = (len(raw) // block) * block
        data = raw[:usable]
        print(f"  Data file  : {args.data_path}")
        print(f"  Full file  : {len(full_raw):,} bytes")
        frac_str = f" ({args.data_frac*100:.0f}%)" if args.data_frac < 1.0 else ""
        print(f"  Train data : {usable:,} bytes{frac_str} (trimmed to batch boundary)")
    else:
        data = make_synthetic_data(size=20_480)
        print(f"  Data       : synthetic ({len(data):,} bytes)")

    # ── 2. Build model ──
    model = HydraBOA(
        d_model=d_model,
        num_layers=args.num_layers,
        vocab_size=256,
        K=K,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params:,}")

    # ── 3. Train ──
    skipped_training = False
    final_bpp = 0.0
    if args.load_checkpoint:
        ckpt_path = Path(args.load_checkpoint)
        if ckpt_path.exists():
            print(f"  [INFO] Loading checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            final_bpp = ckpt.get("final_bpp", 0.0)
            skipped_training = True
        else:
            print(f"  [WARN] Checkpoint not found at {ckpt_path}, will train from scratch.")

    if not skipped_training:
        final_bpp = train_hydra(
            model, data,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            device=device,
            K=K,
            lr=args.lr,
            precision=args.precision,
            use_compile=args.compile,
        )

    # ── 3b. Save checkpoint ──
    if args.save_checkpoint and not skipped_training:
        ckpt_dir = os.path.dirname(args.save_checkpoint)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        ckpt = {
            "model_state_dict": model.state_dict(),
            "d_model": d_model,
            "num_layers": args.num_layers,
            "K": K,
            "vocab_size": 256,
            "final_bpp": final_bpp,
            "epochs": args.epochs,
            "lr": args.lr,
            "data_path": args.data_path,
            "data_frac": args.data_frac,
        }
        torch.save(ckpt, args.save_checkpoint)
        print(f"  Checkpoint saved → {args.save_checkpoint}")

    if args.train_only:
        print("  [INFO] --train-only specified, exiting after training.")
        return 0

    # ── 4. Compress ──
    # When test-bytes=0, compress the FULL file (not just training slice)
    if args.test_bytes == 0 and full_raw is not None:
        compress_source = full_raw
    else:
        compress_source = data
    n_test = args.test_bytes if args.test_bytes > 0 else len(compress_source)
    # Align to K
    n_test = (n_test // K) * K
    test_data = compress_source[:n_test].tobytes()
    assert len(test_data) > 0
    print(f"\n{'─'*60}")
    print(f"  Compress / Decompress test  ({len(test_data)} bytes)")
    print(f"{'─'*60}")

    use_gpu = args.gpu_codec and _HAS_GPU_CODEC and device == "cuda"
    if args.gpu_codec and not use_gpu:
        print("  [WARN] GPU codec not available, falling back to CPU codec")

    t0 = time.perf_counter()
    if use_gpu:
        compressed, metadata = compress_hydra_gpu(
            model, test_data, K=K, device=device, progress=True
        )
        # compressed is a list of uint32 arrays (one per stream)
        comp_bytes = sum(len(a) * 4 for a in compressed)
    else:
        compressed, metadata = compress_hydra(
            model, test_data, K=K, device=device, progress=True
        )
        comp_bytes = len(compressed) * 4  # uint32 words → bytes
    t_comp = time.perf_counter() - t0
    ratio = len(test_data) / comp_bytes if comp_bytes else float("inf")
    comp_mbs = len(test_data) / 1e6 / max(t_comp, 1e-9)
    print(f"  Original   : {len(test_data):,} bytes ({len(test_data)/1e6:.1f} MB)")
    print(f"  Compressed : {comp_bytes:,} bytes  (ratio {ratio:.2f}x)")
    print(f"  Comp. time : {t_comp:.3f}s  ({comp_mbs:.2f} MB/s)")

    # ── 5. Decompress ──
    t0 = time.perf_counter()
    if use_gpu:
        decompressed = decompress_hydra_gpu(
            model, compressed, metadata, device=device, progress=True
        )
    else:
        decompressed = decompress_hydra(
            model, compressed, metadata, device=device, progress=True
        )
    t_dec = time.perf_counter() - t0
    dec_mbs = len(decompressed) / 1e6 / max(t_dec, 1e-9)
    print(f"  Decompressed : {len(decompressed):,} bytes")
    print(f"  Dec. time    : {t_dec:.3f}s  ({dec_mbs:.2f} MB/s)")

    if args.metrics_csv:
        ckpt_stem = Path(args.save_checkpoint).stem if args.save_checkpoint else "hydra_main"
        model_name = args.metrics_model_name or ckpt_stem
        experiment = ""
        if args.save_checkpoint:
            p = Path(args.save_checkpoint)
            experiment = p.parent.parent.parent.name if len(p.parents) >= 3 else p.parent.name
        row = {
            "model": model_name,
            "compression_ratio": f"{ratio:.6f}",
            "throughput_compress_MBps": f"{comp_mbs:.6f}",
            "throughput_decompress_MBps": f"{dec_mbs:.6f}",
            "original_size": str(len(test_data)),
            "compressed_size": str(comp_bytes),
            "time_compress_s": f"{t_comp:.6f}",
            "time_decompress_s": f"{t_dec:.6f}",
            "experiment": experiment,
            "checkpoint_path": str(Path(args.save_checkpoint).resolve()) if args.save_checkpoint else "",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _upsert_metrics_row(Path(args.metrics_csv), row)
        print(f"  [INFO] Updated metrics CSV: {args.metrics_csv}")

    # ── 6. Verify ──
    sha_orig = hashlib.sha256(test_data).hexdigest()
    sha_dec  = hashlib.sha256(decompressed).hexdigest()
    match = test_data == decompressed

    print(f"\n{'═'*60}")
    print(f"  Verification")
    print(f"{'═'*60}")
    print(f"  Original SHA-256     : {sha_orig}")
    print(f"  Decompressed SHA-256 : {sha_dec}")
    if match:
        print(f"  Result: PASS  ✓  (byte-exact round-trip)")
    else:
        print(f"  Result: FAIL  ✗")
        if len(test_data) != len(decompressed):
            print(f"    Length mismatch: {len(test_data)} vs {len(decompressed)}")
        else:
            for i in range(len(test_data)):
                if test_data[i] != decompressed[i]:
                    print(f"    First mismatch at byte {i}: "
                          f"expected {test_data[i]}, got {decompressed[i]}")
                    break

    print()
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())
