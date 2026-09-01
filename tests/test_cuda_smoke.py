"""
cuda_smoke_test.py — CUDA + Model Architecture Smoke Test
==========================================================

Phase 9: Verifies CUDA availability, model instantiation,
and a single forward pass for both legacy and v2 architectures.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

results = {}


def record(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results[name] = {"status": status, "detail": detail}
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return passed


# ─── C01: CUDA availability ───
cuda_ok = torch.cuda.is_available()
record("C01_cuda_available", cuda_ok)

if cuda_ok:
    gpu_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    torch_version = torch.__version__
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    record("C01_gpu_info", True,
           f"{gpu_name}, CUDA {cuda_version}, PyTorch {torch_version}, {allocated:.0f}MB allocated")
else:
    record("C01_gpu_info", False, "No CUDA device found")
    print("\n  WARNING: CUDA not available. Training will fall back to CPU.")
    print("  This is acceptable for testing but not for full runs.\n")

device = torch.device("cuda" if cuda_ok else "cpu")

# ─── C02: Legacy LEGR model instantiation ───
try:
    from src.encoders import LEGRDualEncoder
    model_legacy = LEGRDualEncoder(
        num_tools=45,
        embed_dim=256,
        graph_encoder_type="gcn",
    ).to(device)
    n_params = sum(p.numel() for p in model_legacy.parameters())
    record("C02_legacy_model", True, f"{n_params:,} parameters")
except Exception as e:
    record("C02_legacy_model", False, str(e))
    model_legacy = None

# ─── C03: Legacy forward pass ───
if model_legacy is not None:
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        tokens = tokenizer(["test query"], return_tensors="pt", padding=True, truncation=True)
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        x = torch.tensor([[1]], dtype=torch.long, device=device)
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        batch = torch.tensor([0], dtype=torch.long, device=device)

        model_legacy.eval()
        with torch.no_grad():
            z_text, z_graph = model_legacy(input_ids, attention_mask, x, edge_index, batch)
        record("C03_legacy_forward", True,
               f"z_text={z_text.shape}, z_graph={z_graph.shape}")
    except Exception as e:
        record("C03_legacy_forward", False, str(e))

# ─── C04: V2 model instantiation ───
try:
    from src.encoders_v2 import LEGRDualEncoderV2
    model_v2 = LEGRDualEncoderV2(
        embed_dim=256,
        graph_hidden_dim=128,
        graph_num_layers=3,
        node_feature_dim=64,
    ).to(device)
    n_params_v2 = sum(p.numel() for p in model_v2.parameters())
    record("C04_v2_model", True, f"{n_params_v2:,} parameters")
except Exception as e:
    record("C04_v2_model", False, str(e))
    model_v2 = None

# ─── C05: V2 forward pass with text node features ───
if model_v2 is not None:
    try:
        model_v2.precompute_tool_features(["read_user_profile", "edit_username"], device)
        node_features = model_v2.node_feature_encoder.get_features(
            [["read_user_profile", "edit_username"]], device
        )

        edge_index = torch.tensor([[0], [1]], dtype=torch.long, device=device)
        batch = torch.tensor([0, 0], dtype=torch.long, device=device)

        model_v2.eval()
        with torch.no_grad():
            z_graph = model_v2.encode_graph(node_features, edge_index, batch)
        record("C05_v2_text_features", True,
               f"node_features={node_features.shape}, z_graph={z_graph.shape}")
    except Exception as e:
        record("C05_v2_text_features", False, str(e))

# ─── C06: V2 full forward pass ───
if model_v2 is not None:
    try:
        tokens = tokenizer(["retrieve user profile then edit username"],
                           return_tensors="pt", padding=True, truncation=True)
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        model_v2.eval()
        with torch.no_grad():
            z_text, z_graph = model_v2(
                input_ids, attention_mask,
                node_features, edge_index, batch,
            )
        cos_sim = F.cosine_similarity(z_text, z_graph).item()
        record("C06_v2_full_forward", True,
               f"z_text={z_text.shape}, z_graph={z_graph.shape}, cos_sim={cos_sim:.4f}")
    except Exception as e:
        record("C06_v2_full_forward", False, str(e))

# ─── C07: Directed GNN preserves edge direction ───
if model_v2 is not None:
    try:
        from src.data.tool_registry import get_tools
        tools_3 = get_tools(15)[:3]
        model_v2.precompute_tool_features(tools_3, device)

        # Forward edges: A→B→C
        nf1 = model_v2.node_feature_encoder.get_features([tools_3], device)
        ei1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long, device=device)
        b1 = torch.tensor([0, 0, 0], dtype=torch.long, device=device)

        # Reversed edges: C→B→A
        nf2 = model_v2.node_feature_encoder.get_features([tools_3], device)
        ei2 = torch.tensor([[2, 1], [1, 0]], dtype=torch.long, device=device)
        b2 = torch.tensor([0, 0, 0], dtype=torch.long, device=device)

        model_v2.eval()
        with torch.no_grad():
            z1 = model_v2.encode_graph(nf1, ei1, b1)
            z2 = model_v2.encode_graph(nf2, ei2, b2)

        cos = F.cosine_similarity(z1, z2).item()
        different = cos < 0.999
        record("C07_direction_preserved", different,
               f"forward vs reversed cosine={cos:.4f} (should be < 1.0)")
    except Exception as e:
        record("C07_direction_preserved", False, str(e))

# ─── C08: Loss function ───
try:
    from src.loss import GraphAwareContrastiveLoss
    loss_fn = GraphAwareContrastiveLoss(lambda_ged=0.3)
    z_t = F.normalize(torch.randn(4, 256, device=device), dim=-1)
    z_g = F.normalize(torch.randn(4, 256, device=device), dim=-1)
    ged = torch.tensor([
        [0, 2, 3, 4],
        [2, 0, 1, 3],
        [3, 1, 0, 2],
        [4, 3, 2, 0],
    ], dtype=torch.float, device=device)
    loss, metrics = loss_fn(z_t, z_g, ged)
    record("C08_loss_function", loss.item() > 0,
           f"loss={loss.item():.4f}, r@1={metrics.get('recall_at_1', 'N/A')}")
except Exception as e:
    record("C08_loss_function", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
total = len(results)
passed = sum(1 for v in results.values() if v["status"] == "PASS")
print(f"  CUDA SMOKE TEST: {passed}/{total} passed")
if passed < total:
    failures = [k for k, v in results.items() if v["status"] == "FAIL"]
    print(f"  Failures: {failures}")
print("=" * 60)

report_dir = Path("artifacts/campaign_v4")
report_dir.mkdir(parents=True, exist_ok=True)
report = {
    "device": str(device),
    "cuda_available": cuda_ok,
    "gpu_name": torch.cuda.get_device_name(0) if cuda_ok else None,
    "cuda_version": torch.version.cuda if cuda_ok else None,
    "pytorch_version": torch.__version__,
    "results": results,
}
(report_dir / "cuda_smoke_test.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
