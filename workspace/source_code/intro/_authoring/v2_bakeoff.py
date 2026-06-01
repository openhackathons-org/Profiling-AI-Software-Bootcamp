"""
v2_bakeoff.py — empirical search for a v2 "Bug 2" that satisfies the intro-notebook spec:
  (a) a real perf bug,
  (b) WHERE it is is visible in PyTorch Profiler,
  (c) WHAT it is / how to fix it is NOT obvious from the profiler,
  (d) nsys timeline narrows it down.

Strategy (per session insight): shrink GPU work per step so CPU/launch/sync becomes the
critical path. Then CPU-side concurrency bugs — invisible on GPU-bound ResNet18 — reappear.

This script does NOT use a DataLoader: it holds one synthetic batch resident on the GPU so the
data pipeline is removed as a confound. We measure pure train-step cost for each variant.

Usage:
  python v2_bakeoff.py --model tiny --batch 64
  python v2_bakeoff.py --model resnet18 --batch 64
"""
import argparse, time
import torch, torch.nn as nn


class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, num_classes),
        )
    def forward(self, x): return self.net(x)


def build_model(name):
    if name == "tiny":
        return TinyCNN()
    from torchvision.models import resnet18
    m = resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def run(name, batch, img, iters, warmup, make_opt, step_extra, foreach):
    """Return mean ms/step for one variant."""
    dev = torch.device("cuda")
    torch.manual_seed(0)
    model = build_model(name).to(dev).train()
    opt = make_opt(model)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(batch, 3, img, img, device=dev)
    y = torch.randint(0, 10, (batch,), device=dev)

    times = []
    for i in range(iters):
        if i == warmup:
            torch.cuda.synchronize(); times = []
        t0 = time.perf_counter()
        out = model(x)
        loss = crit(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step_extra(loss, out, y)
        times.append(time.perf_counter())
    torch.cuda.synchronize()
    if len(times) < 2:
        return float("nan")
    return 1000.0 * (times[-1] - times[0]) / (len(times) - 1)


def run_amp(name, batch, img, iters, warmup, pre_backward_extra=None, post_step_extra=None):
    """Like run() but wraps forward+loss in autocast and uses GradScaler.
    pre_backward_extra: called after loss, before backward (serialises CPU before bwd enqueue)
    post_step_extra: called after optimizer step (at end of step)
    """
    dev = torch.device("cuda")
    torch.manual_seed(0)
    model = build_model(name).to(dev).train()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, foreach=True)
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.CrossEntropyLoss()
    x = torch.randn(batch, 3, img, img, device=dev)
    y = torch.randint(0, 10, (batch,), device=dev)

    times = []
    for i in range(iters):
        if i == warmup:
            torch.cuda.synchronize(); times = []
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
            loss = crit(out, y)
        if pre_backward_extra:
            pre_backward_extra(loss, out, y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if post_step_extra:
            post_step_extra(loss, out, y)
        times.append(time.perf_counter())
    torch.cuda.synchronize()
    if len(times) < 2:
        return float("nan")
    return 1000.0 * (times[-1] - times[0]) / (len(times) - 1)


def run_amp_anomaly(name, batch, img, iters, warmup):
    """AMP training with torch.autograd.set_detect_anomaly(True) — left-on debug flag."""
    dev = torch.device("cuda")
    torch.manual_seed(0)
    model = build_model(name).to(dev).train()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, foreach=True)
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.CrossEntropyLoss()
    x = torch.randn(batch, 3, img, img, device=dev)
    y = torch.randint(0, 10, (batch,), device=dev)

    torch.autograd.set_detect_anomaly(True)  # Bug: left-on debug flag
    times = []
    for i in range(iters):
        if i == warmup:
            torch.cuda.synchronize(); times = []
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
            loss = crit(out, y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        times.append(time.perf_counter())
    torch.cuda.synchronize()
    torch.autograd.set_detect_anomaly(False)
    if len(times) < 2:
        return float("nan")
    return 1000.0 * (times[-1] - times[0]) / (len(times) - 1)


def run_amp_no_grad_eval(name, batch, img, iters, warmup, use_no_grad=True):
    """AMP training with an eval pass each step; bug is forgetting torch.no_grad()."""
    dev = torch.device("cuda")
    torch.manual_seed(0)
    model = build_model(name).to(dev).train()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, foreach=True)
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.CrossEntropyLoss()
    x = torch.randn(batch, 3, img, img, device=dev)
    y = torch.randint(0, 10, (batch,), device=dev)
    # separate held-out val batch
    xv = torch.randn(batch, 3, img, img, device=dev)
    yv = torch.randint(0, 10, (batch,), device=dev)

    times = []
    for i in range(iters):
        if i == warmup:
            torch.cuda.synchronize(); times = []
        model.train()
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
            loss = crit(out, y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        # Eval on held-out batch every step
        model.eval()
        if use_no_grad:
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.float16):
                    val_out = model(xv)
        else:
            with torch.autocast("cuda", dtype=torch.float16):
                val_out = model(xv)   # Bug: no torch.no_grad(), builds graph
        times.append(time.perf_counter())
    torch.cuda.synchronize()
    if len(times) < 2:
        return float("nan")
    return 1000.0 * (times[-1] - times[0]) / (len(times) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18", choices=["tiny", "resnet18"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--img", type=int, default=32)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=30)
    args = ap.parse_args()

    noop = lambda loss, out, y: None
    sync_item = lambda loss, out, y: (out.argmax(1) == y).sum().item()

    sgd_fused = lambda m: torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, foreach=True)

    common = dict(name=args.model, batch=args.batch, img=args.img,
                  iters=args.iters, warmup=args.warmup)

    print(f"\n=== model={args.model} batch={args.batch} img={args.img} "
          f"(GPU={torch.cuda.get_device_name(0)}) ===")

    # FP32 baselines
    base_fp32 = run(**common, make_opt=sgd_fused, step_extra=noop, foreach=True)
    print(f"  FP32  clean .......................... {base_fp32:7.3f} ms/step  [{args.batch/(base_fp32/1000):.0f} img/s]")

    fp32_item = run(**common, make_opt=sgd_fused, step_extra=sync_item, foreach=True)
    pct = 100*(fp32_item-base_fp32)/base_fp32
    print(f"  FP32  + per-step .item() ............. {fp32_item:7.3f} ms/step  (Δ {fp32_item-base_fp32:+.3f} ms, {pct:+.1f}%)")

    # AMP baselines
    base_amp = run_amp(**common)
    print(f"  AMP   clean .......................... {base_amp:7.3f} ms/step  [{args.batch/(base_amp/1000):.0f} img/s]")

    # .item() AFTER step (original test position)
    amp_item_post = run_amp(**common, post_step_extra=sync_item)
    pct_post = 100*(amp_item_post-base_amp)/base_amp
    print(f"  AMP   + .item() after step ........... {amp_item_post:7.3f} ms/step  (Δ {amp_item_post-base_amp:+.3f} ms, {pct_post:+.1f}%)")

    # .item() BEFORE backward (the more natural beginner mistake: track loss/acc right after loss)
    amp_item_pre = run_amp(**common, pre_backward_extra=sync_item)
    pct_pre = 100*(amp_item_pre-base_amp)/base_amp
    print(f"  AMP   + .item() before backward ...... {amp_item_pre:7.3f} ms/step  (Δ {amp_item_pre-base_amp:+.3f} ms, {pct_pre:+.1f}%)")

    # TWO .item() calls (loss + accuracy, both inside the loop — very natural mistake)
    two_items = lambda loss, out, y: (loss.item(), (out.argmax(1)==y).float().mean().item())
    amp_item_two = run_amp(**common, pre_backward_extra=two_items)
    pct_two = 100*(amp_item_two-base_amp)/base_amp
    print(f"  AMP   + loss.item()+acc.item() pre-bwd {amp_item_two:7.3f} ms/step  (Δ {amp_item_two-base_amp:+.3f} ms, {pct_two:+.1f}%)")

    # THREE .item() calls: loss + top1_acc + top5_acc (very common real-world logging pattern)
    def three_items(loss, out, y):
        _ = loss.item()
        top1 = (out.argmax(1) == y).float().mean().item()
        top5_preds = out.topk(5, dim=1).indices
        top5 = (top5_preds == y.unsqueeze(1)).any(1).float().mean().item()
    amp_item_three_post = run_amp(**common, post_step_extra=three_items)
    pct_three_post = 100*(amp_item_three_post-base_amp)/base_amp
    print(f"  AMP   + 3×.item() after step ......... {amp_item_three_post:7.3f} ms/step  (Δ {amp_item_three_post-base_amp:+.3f} ms, {pct_three_post:+.1f}%)")

    amp_item_three_pre = run_amp(**common, pre_backward_extra=three_items)
    pct_three_pre = 100*(amp_item_three_pre-base_amp)/base_amp
    print(f"  AMP   + 3×.item() before backward .... {amp_item_three_pre:7.3f} ms/step  (Δ {amp_item_three_pre-base_amp:+.3f} ms, {pct_three_pre:+.1f}%)")

    print(f"\n  AMP speedup (clean):  {base_fp32/base_amp:.2f}x")

    # Anomaly detection left on
    anomaly = run_amp_anomaly(**common)
    pct_an = 100*(anomaly-base_amp)/base_amp
    print(f"\n  AMP + set_detect_anomaly(True) ....... {anomaly:7.3f} ms/step  (Δ {anomaly-base_amp:+.3f} ms, {pct_an:+.1f}%)")

    # Per-step eval: clean (no_grad=True) vs buggy (no_grad=False)
    eval_clean = run_amp_no_grad_eval(**common, use_no_grad=True)
    eval_bug   = run_amp_no_grad_eval(**common, use_no_grad=False)
    overhead_eval = eval_bug - eval_clean
    pct_eval = 100*overhead_eval/eval_clean
    print(f"  AMP + per-step eval (no_grad=True) ... {eval_clean:7.3f} ms/step  (baseline with eval)")
    print(f"  AMP + per-step eval (no_grad=False) .. {eval_bug:7.3f} ms/step  (Δ {overhead_eval:+.3f} ms, {pct_eval:+.1f}% vs eval-clean)")


if __name__ == "__main__":
    main()
