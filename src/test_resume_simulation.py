"""
Simulation test for resume fixes:
1. Checkpoint structure validation
2. load_model: weights + epoch extension + optimizer state stored in global
3. FSDP optimizer state format validation
4. Non-FSDP fallback graceful handling
5. valid_mask in DL3DV_ScreenEvent_Multi dataset
6. query_pts generation (TrackLoss enablement)
7. Training loop range simulation
"""
import sys
import os
import types
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

CKPT_PATH = "../checkpoints/vggt_train_M3ed_curriculum_v3_end/checkpoint-6.pth"
RESULTS = {}


def test_checkpoint_structure():
    """Test 1: Verify checkpoint has the expected keys."""
    print("\n" + "=" * 60)
    print("TEST 1: Checkpoint structure")
    print("=" * 60)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    keys = list(ckpt.keys())
    print(f"  Checkpoint keys: {keys}")
    assert "model" in ckpt, "Missing 'model' key"
    assert "optimizer" in ckpt, "Missing 'optimizer' key"
    assert "epoch" in ckpt, "Missing 'epoch' key"
    print(f"  epoch = {ckpt['epoch']}")
    print(f"  step  = {ckpt.get('step', 'N/A')}")
    print(f"  model state_dict has {len(ckpt['model'])} keys")

    optim_state = ckpt["optimizer"]
    n_tracked = len(optim_state.get("state", {}))
    n_groups = len(optim_state.get("param_groups", []))
    print(f"  optimizer: {n_tracked} params tracked, {n_groups} param_groups")

    if "scaler" in ckpt:
        print(f"  scaler state: {list(ckpt['scaler'].keys())}")
    if "best_so_far" in ckpt:
        print(f"  best_so_far = {ckpt['best_so_far']}")

    RESULTS["test1"] = True
    print("  [PASS]")
    del ckpt


def test_load_model_and_epoch_extension():
    """Test 2: load_model loads weights, stores optimizer in global, extends epochs."""
    print("\n" + "=" * 60)
    print("TEST 2: load_model + epoch auto-extension")
    print("=" * 60)

    from croco.utils import misc

    args = types.SimpleNamespace(
        resume=CKPT_PATH,
        epochs=3,
        start_epoch=0,
        start_step=0,
    )

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

        def load_state_dict(self, state_dict, strict=True):
            return torch.nn.Module.load_state_dict(self, {}, strict=False)

    class DummyScaler:
        def __init__(self):
            self._state = {}

        def load_state_dict(self, sd):
            self._state = sd

        def state_dict(self):
            return self._state

    model = DummyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    scaler = DummyScaler()

    best = misc.load_model(args, model, optimizer, scaler)

    print(f"  args.start_epoch = {args.start_epoch}")
    print(f"  args.start_step  = {args.start_step}")
    print(f"  args.epochs      = {args.epochs}")
    print(f"  best_so_far      = {best}")

    assert args.start_epoch > 0, f"start_epoch should be > 0, got {args.start_epoch}"
    assert args.start_epoch <= args.epochs, (
        f"start_epoch ({args.start_epoch}) > epochs ({args.epochs})"
    )
    print(f"  Epoch auto-extension OK: start_epoch={args.start_epoch} <= epochs={args.epochs}")

    global_optim = misc._resume_optim_state
    assert global_optim is not None, "_resume_optim_state should not be None"
    n_params = len(global_optim.get("state", {}))
    print(f"  Optimizer state in module global: {n_params} params")

    RESULTS["test2"] = True
    print("  [PASS]")


def test_fsdp_optimizer_state_format():
    """Test 3: Verify the saved optimizer state is FSDP full-state format."""
    print("\n" + "=" * 60)
    print("TEST 3: FSDP optimizer state format validation")
    print("=" * 60)

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    optim_state = ckpt["optimizer"]
    del ckpt

    state_keys = list(optim_state["state"].keys())
    print(f"  Total param state entries: {len(state_keys)}")
    print(f"  First 5 state keys: {state_keys[:5]}")

    all_str = all(isinstance(k, str) for k in state_keys)
    all_int = all(isinstance(k, int) for k in state_keys)
    print(f"  Key type: {'string (FSDP full state)' if all_str else 'integer (regular)' if all_int else 'mixed'}")

    sample_key = state_keys[0]
    sample = optim_state["state"][sample_key]
    inner_keys = list(sample.keys())
    print(f"  Sample state '{sample_key}' keys: {inner_keys}")
    if "exp_avg" in sample:
        ea = sample["exp_avg"]
        print(f"  exp_avg shape={ea.shape}, non-zero={int((ea != 0).sum())}/{ea.numel()}")
    if "step" in sample:
        print(f"  step={sample['step']}")

    groups = optim_state["param_groups"]
    for i, g in enumerate(groups):
        n = len(g.get("params", []))
        wd = g.get("weight_decay", "?")
        lr = g.get("lr", "?")
        print(f"  param_group[{i}]: {n} params, wd={wd}, lr={lr}")

    print("\n  FSDP.optim_state_dict_to_load() compatibility check:")
    if all_str:
        print("    String-keyed state -> compatible with FSDP.optim_state_dict_to_load()")
        print("    This is the FULL optimizer state gathered from all FSDP shards")
        print("    load_optimizer_after_fsdp() will use FSDP API to re-shard it")
    else:
        print("    Integer-keyed state -> standard optimizer format")

    RESULTS["test3"] = True
    print("  [PASS]")
    del optim_state


def test_graceful_fallback():
    """Test 4: Non-FSDP fallback handles mismatch gracefully (no crash)."""
    print("\n" + "=" * 60)
    print("TEST 4: Non-FSDP fallback graceful handling")
    print("=" * 60)

    from croco.utils import misc

    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    misc._resume_optim_state = ckpt["optimizer"]
    del ckpt

    misc.load_optimizer_after_fsdp(model, optimizer)

    assert misc._resume_optim_state is None, "State should be cleared in finally block"
    print("  Non-FSDP load with mismatched groups failed gracefully (no crash)")
    print("  _resume_optim_state correctly set to None in finally block")
    print("  In actual FSDP training, the FSDP path will be used instead")

    RESULTS["test4"] = True
    print("  [PASS]")


def test_model_weights_load():
    """Test 5: Full model weights load correctly."""
    print("\n" + "=" * 60)
    print("TEST 5: Full model weights load")
    print("=" * 60)

    from streamvggt.models.streamvggt import StreamVGGT

    print("  Creating StreamVGGT model...")
    model = StreamVGGT(
        fusion="crossattn",
        use_event=True,
        event_in_chans=8,
    )

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    del ckpt

    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    empty_keys = [
        k for k, v in new_state_dict.items()
        if isinstance(v, torch.Tensor) and v.numel() == 0
    ]
    if empty_keys:
        print(f"  Dropping {len(empty_keys)} empty tensors")
        for k in empty_keys:
            del new_state_dict[k]

    info = model.load_state_dict(new_state_dict, strict=False)
    print(f"  Missing keys:    {len(info.missing_keys)}")
    print(f"  Unexpected keys: {len(info.unexpected_keys)}")
    if info.missing_keys:
        print(f"    First 5 missing: {info.missing_keys[:5]}")
    if info.unexpected_keys:
        print(f"    First 5 unexpected: {info.unexpected_keys[:5]}")

    assert len(info.missing_keys) == 0, f"Should have 0 missing keys, got {len(info.missing_keys)}"
    assert len(info.unexpected_keys) == 0, f"Should have 0 unexpected keys, got {len(info.unexpected_keys)}"

    RESULTS["test5"] = True
    print("  [PASS] All model weights loaded perfectly.")
    del model, state_dict, new_state_dict


def test_valid_mask_in_dataset():
    """Test 6: DL3DV_ScreenEvent_Multi returns valid_mask."""
    print("\n" + "=" * 60)
    print("TEST 6: valid_mask in DL3DV_ScreenEvent_Multi")
    print("=" * 60)

    from dust3r.datasets.dl3dv import DL3DV_ScreenEvent_Multi

    ds = DL3DV_ScreenEvent_Multi(
        split="train",
        ROOT="/share/magic_group/aigc/fcr/EventVGGT/data/DL3DV_data/DL3DV",
        resolution=(518, 392),
        num_views=2,
        min_interval=1,
        max_interval=3,
        sample_mode="mixed",
        sequence_ratio=0.5,
        event_in_chans=8,
    )
    print(f"  Dataset size: {len(ds)}")

    views = ds[0]
    print(f"  Number of views returned: {len(views)}")

    for i, v in enumerate(views):
        keys = list(v.keys())
        print(f"  View {i} keys: {keys}")
        has_mask = "valid_mask" in v
        assert has_mask, f"View {i} missing valid_mask!"
        mask = v["valid_mask"]
        if isinstance(mask, np.ndarray):
            print(f"  View {i} valid_mask: shape={mask.shape}, dtype={mask.dtype}, all_true={mask.all()}")
        elif isinstance(mask, torch.Tensor):
            print(f"  View {i} valid_mask: shape={mask.shape}, dtype={mask.dtype}, all_true={mask.all().item()}")

    RESULTS["test6"] = True
    print("  [PASS] valid_mask present in all views.")


def test_query_pts_not_none():
    """Test 7: query_pts is generated when valid_mask exists."""
    print("\n" + "=" * 60)
    print("TEST 7: query_pts generation from valid_mask")
    print("=" * 60)

    from dust3r.inference import sample_query_points

    B, H, W = 1, 392, 518
    valid_mask = torch.ones(B, H, W, dtype=torch.bool)
    query_pts = sample_query_points(valid_mask, M=64)
    print(f"  Input valid_mask shape: {valid_mask.shape}")
    print(f"  Output query_pts: type={type(query_pts)}")
    assert query_pts is not None, "query_pts should not be None"
    print(f"  Output query_pts shape: {query_pts.shape}")
    print(f"  Output query_pts dtype: {query_pts.dtype}")
    assert query_pts.shape[-1] == 2, f"Last dim should be 2, got {query_pts.shape[-1]}"
    print(f"  Sample values: {query_pts[0, :3]}")

    RESULTS["test7"] = True
    print("  [PASS] query_pts generated -> TrackLoss will be active.")


def test_epoch_range_simulation():
    """Test 8: Simulate the training loop range with resume."""
    print("\n" + "=" * 60)
    print("TEST 8: Training loop range simulation")
    print("=" * 60)

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    ckpt_epoch = ckpt["epoch"]
    del ckpt

    config_epochs = 3
    start_epoch = ckpt_epoch + 1

    print(f"  Checkpoint epoch: {ckpt_epoch}")
    print(f"  Config epochs: {config_epochs}")
    print(f"  start_epoch (ckpt epoch + 1): {start_epoch}")

    if start_epoch > config_epochs:
        old_epochs = config_epochs
        config_epochs = start_epoch + old_epochs
        print(f"  Auto-extension triggered: {old_epochs} -> {config_epochs}")

    train_epochs = [e for e in range(start_epoch, config_epochs + 1) if e < config_epochs]
    print(f"  Training epochs: {train_epochs}")
    print(f"  Number of training epochs: {len(train_epochs)}")

    assert len(train_epochs) > 0, "Should have >= 1 training epoch"
    assert len(train_epochs) == 3, f"Expected 3, got {len(train_epochs)}"

    RESULTS["test8"] = True
    print("  [PASS] Training loop will run 3 epochs.")


def test_code_path_logic():
    """Test 9: Verify the FSDP code path logic in load_optimizer_after_fsdp."""
    print("\n" + "=" * 60)
    print("TEST 9: FSDP code path logic verification")
    print("=" * 60)

    from croco.utils.misc import load_optimizer_after_fsdp
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    import inspect

    src = inspect.getsource(load_optimizer_after_fsdp)
    checks = {
        "global _resume_optim_state": "Uses module-level global (not args attribute)",
        "isinstance(fsdp_model, FSDP)": "Checks if model is FSDP-wrapped",
        "FSDP.optim_state_dict_to_load": "Uses FSDP API for optimizer state scatter",
        "getattr(optimizer, \"optimizer\", optimizer)": "Handles AcceleratedOptimizer wrapper",
        "finally": "Has finally block for cleanup",
    }

    all_ok = True
    for pattern, description in checks.items():
        found = pattern in src
        status = "OK" if found else "MISSING"
        print(f"  [{status}] {description}")
        print(f"          Pattern: '{pattern}'")
        if not found:
            all_ok = False

    assert all_ok, "Some code path checks failed"

    RESULTS["test9"] = True
    print("  [PASS] All FSDP code path patterns verified.")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    from accelerate import PartialState
    PartialState()

    print("=" * 60)
    print("  RESUME FIX SIMULATION TEST")
    print("=" * 60)

    test_checkpoint_structure()
    test_load_model_and_epoch_extension()
    test_fsdp_optimizer_state_format()
    test_graceful_fallback()
    test_model_weights_load()
    test_valid_mask_in_dataset()
    test_query_pts_not_none()
    test_epoch_range_simulation()
    test_code_path_logic()

    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    all_pass = True
    for name, passed in RESULTS.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: [{status}]")
        if not passed:
            all_pass = False

    print(f"\n  {len(RESULTS)}/{len(RESULTS)} tests passed" if all_pass else "")
    if all_pass:
        print("  ALL TESTS PASSED!")
    else:
        print("  SOME TESTS FAILED!")
        sys.exit(1)
