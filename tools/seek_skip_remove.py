from pathlib import Path

DIR_PATH = Path(__file__).resolve().parent
TEST_LOG_PATH = DIR_PATH.parent / "tester/api_config/test_log"
TEST_LOG_PATH.mkdir(parents=True, exist_ok=True)

LOG_PREFIXES = {
    "checkpoint": "checkpoint",
    "pass": "api_config_pass",
    "paddle_error": "api_config_paddle_error",
    "torch_error": "api_config_torch_error",
    "paddle_to_torch_failed": "api_config_paddle_to_torch_failed",
    "accuracy_error": "api_config_accuracy_error",
    "timeout": "api_config_timeout",
    "crash": "api_config_crash",
}

log_counts = {}
api_configs = set()
checkpoint_file = TEST_LOG_PATH / "checkpoint.txt"
if not checkpoint_file.exists():
    exit(0)
try:
    with checkpoint_file.open("r") as f:
        api_configs = set(line.strip() for line in f if line.strip())
        log_counts["checkpoint"] = len(api_configs)
except Exception as err:
    print(f"Error reading {checkpoint_file}: {err}", flush=True)

for log_type, prefix in LOG_PREFIXES.items():
    if log_type == "checkpoint":
        continue
    log_file = TEST_LOG_PATH / f"{prefix}.txt"
    if not log_file.exists():
        continue
    try:
        with log_file.open("r") as f:
            lines = set(line.strip() for line in f if line.strip())
            api_configs -= lines
            log_counts[log_type] = len(lines)
    except Exception as err:
        print(f"Error reading {log_file}: {err}", flush=True)

if api_configs:
    log_counts["skip"] = len(api_configs)
    skip_file = TEST_LOG_PATH / "api_config_skip.txt"
    try:
        with skip_file.open("w") as f:
            f.writelines(f"{line}\n" for line in sorted(api_configs))
    except Exception as err:
        print(f"Error writing to {skip_file}: {err}", flush=True)

for log_type, count in log_counts.items():
    print(f"{log_type}: {count}", flush=True)

if api_configs:
    try:
        with checkpoint_file.open("r") as f:
            lines = set(line.strip() for line in f if line.strip())
            lines -= api_configs
            print(f"checkpoint remaining: {len(lines)}", flush=True)
        with checkpoint_file.open("w") as f:
            f.writelines(f"{line}\n" for line in sorted(lines))
    except Exception as err:
        print(f"Error reading {checkpoint_file}: {err}", flush=True)
