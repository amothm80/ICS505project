"""Train every dataset under data/ using the same pipeline as train.sh.

Each dataset is trained via torchrun + main_finetune.py with the settings from
train.sh. After each run, the details and final test metrics are appended to
artifacts/runs.csv.

Usage:
    python train_all.py --mode lora --epochs 50
    python train_all.py --mode finetune --epochs 20
    python train_all.py --mode lora --epochs 100 --blr 2e-2 --layer_decay 1.0 \
        --lora_rank 16 --datasets OCTID PAPILA Retina JSIEC

Optimizer flags left unset are not passed, so main_finetune.py defaults apply.
"""
import argparse
import csv
import datetime
import json
import os
import re
import subprocess
import sys
import time

# ==== Model settings (mirrors train.sh) ====
MODEL = "RETFound_mae"
MODEL_ARCH = "retfound_mae"
FINETUNE = "RETFound_mae_natureCFP"

BATCH_SIZE = 24
INPUT_SIZE = 224
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
MASTER_PORT = 48766

OUTPUT_DIR = "./output_dir"
METRIC_COLS = ["val_loss", "accuracy", "f1", "roc_auc", "hamming", "jaccard",
               "precision", "recall", "average_precision", "kappa"]


def get_args():
    parser = argparse.ArgumentParser("Train all datasets in the data folder")
    parser.add_argument("--mode", required=True, choices=["finetune", "lora"],
                        help="finetune = full fine-tuning, lora = LoRA adapters")
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--artifacts_dir", default="artifacts")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="only train these dataset folder names (default: all)")
    # LoRA shape
    parser.add_argument("--lora_rank", type=int, default=LORA_RANK)
    parser.add_argument("--lora_alpha", type=float, default=None,
                        help="default: 2 * lora_rank")
    parser.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT)
    # Optimizer: None = don't pass, keep main_finetune.py default
    parser.add_argument("--blr", type=float, default=None,
                        help="base LR; actual lr = blr * batch_size / 256")
    parser.add_argument("--layer_decay", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tag", default="",
                        help="extra suffix for the output dir / log name")
    args = parser.parse_args()
    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_rank
    return args


OPT_FLAGS = ["blr", "layer_decay", "weight_decay", "warmup_epochs", "seed"]


def config_suffix(args):
    """Unique name per config so runs don't overwrite each other's checkpoints."""
    parts = [args.mode, f"e{args.epochs}"]
    if args.mode == "lora":
        parts.append(f"r{args.lora_rank}a{args.lora_alpha:g}")
    for k in OPT_FLAGS:
        v = getattr(args, k)
        if v is not None:
            parts.append(f"{k}{v:g}")
    if args.tag:
        parts.append(args.tag)
    return "_".join(parts)


def count_classes(dataset_path):
    train_dir = os.path.join(dataset_path, "train")
    return len([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])


def col_name(prefix, metric):
    # metrics_*.csv calls the loss "val_loss" for both splits
    return f"{prefix}_loss" if metric == "val_loss" else f"{prefix}_{metric}"


def read_csv_rows(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def read_log_lines(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
        return out.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return ""


def run_dataset(dataset, num_class, task, args, log_path):
    cmd = [
        "torchrun", "--nproc_per_node=1", f"--master_port={MASTER_PORT}", "main_finetune.py",
        "--model", MODEL,
        "--model_arch", MODEL_ARCH,
        "--finetune", FINETUNE,
        "--savemodel",
        "--global_pool",
        "--batch_size", str(BATCH_SIZE),
        "--world_size", "1",
        "--epochs", str(args.epochs),
        "--nb_classes", str(num_class),
        "--data_path", f"./{dataset}",
        "--input_size", str(INPUT_SIZE),
        "--task", task,
        "--adaptation", args.mode,
        "--lora_rank", str(args.lora_rank),
        "--lora_alpha", f"{args.lora_alpha:g}",
        "--lora_dropout", str(args.lora_dropout),
    ]
    for k in OPT_FLAGS:
        v = getattr(args, k)
        if v is not None:
            cmd += [f"--{k}", f"{v:g}" if isinstance(v, float) else str(v)]

    # Values parsed from main_finetune.py stdout
    stats = {"best_epoch": "", "best_val_score": "", "train_time": "",
             "max_gpu_mem_mb": 0.0, "total_params": "", "trainable_params": "",
             "trainable_pct": "", "lr": "", "blr_used": "", "layer_decay_used": ""}
    num = r"([\d.]+(?:[eE][+-]?\d+)?)"
    patterns = {
        # MAE/RETFound print "base lr: 5.00e-03" and "actual lr: 4.69e-04" at startup,
        # plus the args Namespace ("blr=0.005", "layer_decay=0.65", ...)
        "lr": re.compile(r"actual lr:\s*" + num),
        "blr": re.compile(r"(?:base lr:\s*|\bblr=)" + num),
        "ld": re.compile(r"\blayer_decay=" + num),
        "best": re.compile(r"Best epoch = (\d+), Best score = ([\d.]+)"),
        "mem": re.compile(r"max mem: ([\d.]+)"),
        "time": re.compile(r"Training time (\S+)"),
        "total": re.compile(r"Total parameters:\s+([\d,]+)"),
        "trainable": re.compile(r"Trainable parameters:\s+([\d,]+)"),
        "pct": re.compile(r"Trainable percentage:\s+([\d.]+)%"),
    }

    # Stream output to the console and a per-run log file
    with open(log_path, "w", encoding="utf8") as log_f:
        log_f.write(" ".join(cmd) + "\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log_f.write(line)
            if m := patterns["best"].search(line):
                stats["best_epoch"], stats["best_val_score"] = int(m.group(1)), float(m.group(2))
            if m := patterns["mem"].search(line):
                stats["max_gpu_mem_mb"] = max(stats["max_gpu_mem_mb"], float(m.group(1)))
            if m := patterns["time"].search(line):
                stats["train_time"] = m.group(1)
            if m := patterns["total"].search(line):
                stats["total_params"] = int(m.group(1).replace(",", ""))
            if m := patterns["trainable"].search(line):
                stats["trainable_params"] = int(m.group(1).replace(",", ""))
            if m := patterns["pct"].search(line):
                stats["trainable_pct"] = float(m.group(1))
            if stats["lr"] == "" and (m := patterns["lr"].search(line)):
                stats["lr"] = float(m.group(1))
            if stats["blr_used"] == "" and (m := patterns["blr"].search(line)):
                stats["blr_used"] = float(m.group(1))
            if stats["layer_decay_used"] == "" and (m := patterns["ld"].search(line)):
                stats["layer_decay_used"] = float(m.group(1))
        proc.wait()

    # Fall back to computing the values if main_finetune.py didn't print them
    if stats["blr_used"] == "" and args.blr is not None:
        stats["blr_used"] = args.blr
    if stats["lr"] == "" and stats["blr_used"] != "":
        stats["lr"] = stats["blr_used"] * BATCH_SIZE / 256

    return proc.returncode, stats


def migrate_csv(path, fieldnames):
    """If runs.csv has an older header, rewrite it with the new columns (blank for old rows).

    Without this, DictWriter would append rows whose columns don't line up with the header.
    """
    if not os.path.isfile(path):
        return
    with open(path, newline="", encoding="utf8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == fieldnames:
            return
        rows = list(reader)
    backup = path + ".bak"
    os.replace(path, backup)
    with open(path, "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Upgraded {path} to the new column layout (backup: {backup})")


def main():
    args = get_args()
    logs_dir = os.path.join(args.artifacts_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    runs_csv = os.path.join(args.artifacts_dir, "runs.csv")

    datasets = sorted(
        os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d, "train"))
    )
    if args.datasets:
        missing = set(args.datasets) - {os.path.basename(d) for d in datasets}
        if missing:
            sys.exit(f"Unknown dataset(s): {', '.join(sorted(missing))}")
        datasets = [d for d in datasets if os.path.basename(d) in args.datasets]
    print(f"Found {len(datasets)} datasets: {', '.join(datasets)}")
    suffix = config_suffix(args)

    gpu_name = get_gpu_name()
    fieldnames = (
        ["timestamp", "dataset", "model", "model_arch", "finetune", "adaptation", "epochs",
         "nb_classes", "batch_size", "input_size", "lora_rank", "lora_alpha", "lora_dropout",
         "blr", "lr", "layer_decay", "weight_decay", "warmup_epochs", "seed", "tag",
         "total_params", "trainable_params", "trainable_pct", "gpu", "status",
         "wall_time_sec", "train_time", "max_gpu_mem_mb", "best_epoch", "best_val_score",
         "final_train_loss", "final_train_lr"]
        + [col_name("best_val", c) for c in METRIC_COLS]
        + [col_name("test", c) for c in METRIC_COLS]
        + ["test_score", "output_dir", "log_file"]
    )

    migrate_csv(runs_csv, fieldnames)

    for dataset in datasets:
        num_class = count_classes(dataset)
        name = os.path.basename(dataset)
        # one dir per dataset+config; old runs used f"{MODEL_ARCH}_{dataset}_{mode}"
        task = f"{MODEL_ARCH}_{name}_{suffix}"
        task_dir = os.path.join(OUTPUT_DIR, task)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(logs_dir, f"{name}_{suffix}_{stamp}.log")
        print(f"\n===== {dataset} | classes={num_class} | mode={args.mode} | epochs={args.epochs} =====")

        # main_finetune.py appends to these files, so remember where this run starts
        val_csv = os.path.join(task_dir, "metrics_val.csv")
        test_csv = os.path.join(task_dir, "metrics_test.csv")
        train_log = os.path.join(task_dir, "log.txt")
        n_val, n_test, n_log = len(read_csv_rows(val_csv)), len(read_csv_rows(test_csv)), len(read_log_lines(train_log))

        start = time.time()
        returncode, stats = run_dataset(dataset, num_class, task, args, log_path)
        wall_time = int(time.time() - start)

        val_rows = read_csv_rows(val_csv)[n_val:]
        test_rows = read_csv_rows(test_csv)[n_test:]
        log_rows = read_log_lines(train_log)[n_log:]

        # One val row per epoch, so the best epoch indexes into this run's rows
        best_val = {}
        if stats["best_epoch"] != "" and stats["best_epoch"] < len(val_rows):
            best_val = val_rows[stats["best_epoch"]]
        test = test_rows[-1] if returncode == 0 and test_rows else {}
        last_log = log_rows[-1] if log_rows else {}

        test_score = ""
        if test:
            test_score = (float(test["f1"]) + float(test["roc_auc"]) + float(test["kappa"])) / 3

        row = {
            "timestamp": stamp,
            "dataset": name,
            "model": MODEL,
            "model_arch": MODEL_ARCH,
            "finetune": FINETUNE,
            "adaptation": args.mode,
            "epochs": args.epochs,
            "nb_classes": num_class,
            "batch_size": BATCH_SIZE,
            "input_size": INPUT_SIZE,
            "lora_rank": args.lora_rank if args.mode == "lora" else "",
            "lora_alpha": f"{args.lora_alpha:g}" if args.mode == "lora" else "",
            "lora_dropout": args.lora_dropout if args.mode == "lora" else "",
            "blr": stats["blr_used"] if stats["blr_used"] != "" else "default",
            "lr": stats["lr"],
            "layer_decay": (args.layer_decay if args.layer_decay is not None
                            else stats["layer_decay_used"] or "default"),
            "weight_decay": args.weight_decay if args.weight_decay is not None else "default",
            "warmup_epochs": args.warmup_epochs if args.warmup_epochs is not None else "default",
            "seed": args.seed if args.seed is not None else "default",
            "tag": args.tag,
            "total_params": stats["total_params"],
            "trainable_params": stats["trainable_params"],
            "trainable_pct": stats["trainable_pct"],
            "gpu": gpu_name,
            "status": "success" if returncode == 0 else f"failed ({returncode})",
            "wall_time_sec": wall_time,
            "train_time": stats["train_time"],
            "max_gpu_mem_mb": stats["max_gpu_mem_mb"] or "",
            "best_epoch": stats["best_epoch"],
            "best_val_score": stats["best_val_score"],
            "final_train_loss": last_log.get("train_loss", ""),
            "final_train_lr": last_log.get("train_lr", ""),
            **{col_name("best_val", c): best_val.get(c, "") for c in METRIC_COLS},
            **{col_name("test", c): test.get(c, "") for c in METRIC_COLS},
            "test_score": test_score,
            "output_dir": task_dir,
            "log_file": log_path,
        }

        # Append after every dataset so partial progress is kept
        write_header = not os.path.isfile(runs_csv)
        with open(runs_csv, "a", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"Recorded run for {name} in {runs_csv} ({row['status']})")


if __name__ == "__main__":
    main()
