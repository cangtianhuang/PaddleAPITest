import os
import shutil
from pathlib import Path

# 日志文件路径
DIR_PATH = Path(__file__).resolve()
while DIR_PATH.name != "PaddleAPITest":
    DIR_PATH = DIR_PATH.parent
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
}

is_engineV2 = False

    
# Command line arguments configuration
CMD_CONFIG = None

def get_cfg():
    global CMD_CONFIG
    return CMD_CONFIG

def set_cfg(cfg):
    global CMD_CONFIG
    if cfg.id != "":
        cfg.id = "_"+cfg.id
    CMD_CONFIG = cfg

def set_engineV2():
    global is_engineV2
    is_engineV2 = True
    TMP_LOG_PATH.mkdir(exist_ok=True)


def get_log_file(log_type: str):
    """获取指定日志类型和PID对应的日志文件路径"""
    global is_engineV2
    prefix = LOG_PREFIXES.get(log_type)
    id = get_cfg().id
    if not is_engineV2:
        return TEST_LOG_PATH / f"{prefix + id}.txt"
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
    id = get_cfg().id
    file_path = TEST_LOG_PATH / f"{LOG_PREFIXES[log_type] + id}.txt"
    try:
        with file_path.open("r") as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return set()
    except Exception as err:
        print(f"Error reading {file_path}: {err}", flush=True)
        return set()


def aggregate_logs(mkdir=False):
    """聚合所有相同类型的日志文件"""
    if not TMP_LOG_PATH.exists():
        if mkdir:
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
    if mkdir:
        TMP_LOG_PATH.mkdir(exist_ok=True)

def print_log_info(all_case, fail_case):
    """打印日志统计信息"""
    log_counts = {}
    recorded_count = 0
    
    for log_type, prefix in LOG_PREFIXES.items():
        log_file = TEST_LOG_PATH / f"{prefix}.txt"
        if not log_file.exists():
            continue
        try:
            with log_file.open("r") as f:
                count = sum(1 for _ in f)
                log_counts[log_type] = count
                if log_type not in ["checkpoint", "timeout", "crash"]:
                    recorded_count += count
        except Exception as err:
            print(f"Error reading {log_file}: {err}", flush=True)

    skipped_case = all_case - recorded_count - fail_case
    assert skipped_case >= 0, f"skipped_case should be non-negative, but got {skipped_case}"
    
    # 打印统计信息
    print("\n" + "="*50)
    print("Test Case Statistics".center(50))
    print("="*50)
    print(f"{'Total cases':<30}: {all_case}")
    print(f"{'Failed cases':<30}: {fail_case}")
    print(f"{'Skipped cases':<30}: {skipped_case}")
    print("-"*50)
    print("Log Type Breakdown:")
    for log_type, count in log_counts.items():
        print(f"  {log_type:<28}: {count}")
    print("="*50 + "\n")
