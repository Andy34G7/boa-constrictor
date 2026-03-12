import torch
import torch.nn as nn
from datetime import datetime
from tqdm.auto import tqdm
import numpy as np

# --- eval loop (reports mean bpp across the loader) ---
@torch.inference_mode()
def evaluate_bpp(model, loader, criterion, device="cuda", vocab_size=256):
    model.eval().to(device)
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        x = batch[:, :-1].to(device)
        y = batch[:, 1:].to(device)
        logits = model(x)  # [B, L-1, vocab_size]
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    mean_nll = total_loss / max(1, total_tokens)
    bpp = mean_nll / np.log(2)  # bits per input byte
    return bpp

def train(model, train_loader, val_loader, test_loader, optimizer, criterion, device="cuda", name="BoaBytePredictor", NUM_EPOCHS=10, PRECISION="fp32", progress=True, start_epoch=1, vocab_size=256):
    
    IS_CUDA = torch.cuda.is_available() and device == "cuda"
    
    print(f"[INFO] Using precision = {PRECISION}")

    def get_autocast_dtype(precision):
        if precision == "fp16":
            return torch.float16
        elif precision == "fp8":
            try:
                return torch.float8_e5m2  # Hopper architecture only (H100 / RTX 5090)
            except AttributeError:
                print("[WARN] FP8 not supported on this PyTorch build, falling back to FP16")
                return torch.float16
        else:
            return torch.float32
        
    AUTODTYPE = get_autocast_dtype(PRECISION)
    amp_enabled = PRECISION in ["fp16", "fp8"] and IS_CUDA
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_steps_per_epoch = len(train_loader)
    total_train_steps = max(1, train_steps_per_epoch)
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train().to(device)
        loader = tqdm(train_loader, total=total_train_steps, desc=f"Epoch {epoch} [{PRECISION}]", disable=not progress)
        for batch in loader:
            x = batch[:, :-1].to(device, non_blocking=True)
            y = batch[:, 1:].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # --- Automatic mixed precision block ---
            with torch.autocast(device_type=device, dtype=AUTODTYPE, enabled=amp_enabled):
                logits = model(x)
                loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

            # --- Scaled backward for FP16/FP8 ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # --- Progress ---
            bits_per_byte = loss.item() / np.log(2)
            if progress:
                loader.set_postfix(
                    loss=f"{loss.item():.4f}",
                    bits=f"{bits_per_byte:.3f}",
                    ratio=f"{(8 / bits_per_byte):.2f}x"
                )

        torch.save(model.state_dict(), f"{name}_{datetime.now().strftime('%dth%b')}_Checkpoint_epoch_{epoch}_{PRECISION}.pt")
        val_bpp = evaluate_bpp(model, val_loader, criterion, device=device, vocab_size=vocab_size)
        print(f"[Epoch {epoch}] val bpp={val_bpp:.4f} (ratio ~ {8/val_bpp:.2f}x)")

    torch.save(model.state_dict(), f"{name}_final_model_{PRECISION}.pt")
    test_bpp = evaluate_bpp(model, test_loader, criterion, device=device, vocab_size=vocab_size)
    print(f"[TEST] bpp={test_bpp:.4f}  ratio ~ {8/test_bpp:.2f}x")

