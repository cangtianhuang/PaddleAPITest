import os
import shutil
from pathlib import Path

# 日志文件路径
DIR_PATH = Path(__file__).resolve()
DIR_PATH = DIR_PATH.parent.parent.parent
TEST_LOG_PATH = DIR_PATH / "tester/api_config/test_log"
TEST_LOG_PATH.mkdir(parents=True, exist_ok=True)
TMP_LOG_PATH = TEST_LOG_PATH / ".tmp"

# 日志类型和对应的文件
LOG_PREFIXES = {
    "checkpoint": "checkpoint",
    "pass": "api_config_pass",
    "paddle_error": "api_config_paddle_error",
    "torch_error": "api_config_torch_error",
    "paddle_to_torch_failed": "api_config_paddle_to_torch_failed",
    "accuracy_error": "api_config_accuracy_error",
    "timeout": "api_config_timeout",
    "crash": "api_config_crash",
    "oom": "api_config_oom",
}

is_engineV2 = False


# Command line arguments configuration
# Used in engine.py
CMD_CONFIG = None


def get_cfg():
    global CMD_CONFIG
    return CMD_CONFIG


def set_cfg(cfg):
    global CMD_CONFIG
    if cfg.id != "":
        cfg.id = "_" + cfg.id
    CMD_CONFIG = cfg


def set_engineV2():
    global is_engineV2
    is_engineV2 = True
    TMP_LOG_PATH.mkdir(exist_ok=True)


def get_log_file(log_type: str):
    """获取指定日志类型和PID对应的日志文件路径"""
    global is_engineV2
    prefix = LOG_PREFIXES.get(log_type)
    if not is_engineV2:
        cfg = get_cfg()
        if cfg:
            return TEST_LOG_PATH / f"{prefix + cfg.id}.txt"
        else:
            return TEST_LOG_PATH / f"{prefix}.txt"
    pid = os.getpid()
    return TMP_LOG_PATH / f"{prefix}_{pid}.txt"


def write_to_log(log_type, line):
    """添加单条日志到当前进程的日志文件"""
    if log_type not in LOG_PREFIXES:
        raise ValueError(f"Invalid log type: {log_type}")
    line = line.strip()
    if not line:
        return
    file_path = get_log_file(log_type)
    try:
        with file_path.open("a") as f:
            f.write(line + "\n")
    except Exception as err:
        print(f"Error writing to {file_path}: {err}", flush=True)


def read_log(log_type):
    """读取文件所有行，返回集合"""
    if log_type not in LOG_PREFIXES:
        raise ValueError(f"Invalid log type: {log_type}")
    cfg = get_cfg()
    if cfg:
        file_path = TEST_LOG_PATH / f"{LOG_PREFIXES[log_type] + cfg.id}.txt"
    else:
        file_path = TEST_LOG_PATH / f"{LOG_PREFIXES[log_type]}.txt"
    try:
        with file_path.open("r") as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return set()
    except Exception as err:
        print(f"Error reading {file_path}: {err}", flush=True)
        return set()


def aggregate_logs(end=False):
    """聚合所有相同类型的日志文件"""
    if not TMP_LOG_PATH.exists() and not end:
        TMP_LOG_PATH.mkdir(exist_ok=True)
        return

    for prefix in LOG_PREFIXES.values():
        log_files = list(TMP_LOG_PATH.glob(f"{prefix}_*.txt"))
        if not log_files:
            continue

        all_lines = set()
        for file_path in log_files:
            try:
                with file_path.open("r") as f:
                    all_lines.update(line.strip() for line in f if line.strip())
            except Exception as err:
                print(f"Error reading {file_path}: {err}", flush=True)

        aggregated_file = TEST_LOG_PATH / f"{prefix}.txt"
        try:
            with aggregated_file.open("a") as f:
                f.writelines(f"{line}\n" for line in sorted(all_lines))
        except Exception as err:
            print(f"Error writing to {aggregated_file}: {err}", flush=True)

    try:
        shutil.rmtree(TMP_LOG_PATH)
    except OSError:
        pass

    if not end:
        TMP_LOG_PATH.mkdir(exist_ok=True)
    else:
        log_counts = {}
        checkpoint_file = TEST_LOG_PATH / "checkpoint.txt"
        api_configs = set()
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
        return log_counts


def print_log_info(all_case, log_counts={}):
    """打印日志统计信息"""
    test_case = log_counts.get("checkpoint", 0)
    fail_case = log_counts.get("crash", 0) + log_counts.get("timeout", 0)
    skip_case = log_counts.get("skip", 0)

    # 打印统计信息
    print("\n" + "=" * 50)
    print("Test Case Statistics".center(50))
    print("=" * 50)
    print(f"{'Total cases':<30}: {all_case}")
    print(f"{'Tested cases':<30}: {test_case}")
    print(f"{'Failed cases':<30}: {fail_case}")
    print(f"{'Skipped cases':<30}: {skip_case}")
    if log_counts:
        print("-" * 50)
        print("Log Type Breakdown:")
        for log_type, count in log_counts.items():
            print(f"  {log_type:<28}: {count}")
    print("=" * 50 + "\n")
