import argparse
import errno
import math
import os
import signal
import sys
import time
from concurrent.futures import TimeoutError, as_completed
from datetime import datetime
from multiprocessing import Lock, Manager, cpu_count, set_start_method
from typing import TYPE_CHECKING

import pynvml
from pebble import ProcessPool

if TYPE_CHECKING:
    from tester import (
        APIConfig,
        APITestAccuracy,
        APITestCINNVSDygraph,
        APITestPaddleOnly,
    )

from tester.api_config.log_writer import (aggregate_logs, print_log_info,
                                          read_log, set_engineV2, write_to_log)


def cleanup(pool):
    print(f"{datetime.now()} Cleanup started", flush=True)
    if pool is not None:
        try:
            if pool.active:
                pool.stop()
                pool.join(timeout=5)
        except Exception as e:
            print(f"{datetime.now()} Error shutting down executor: {e}", flush=True)
    print(f"{datetime.now()} Cleanup completed", flush=True)


def estimate_timeout(api_config) -> float:
    """Estimate timeout based on tensor size in APIConfig."""
    TIMEOUT_STEPS = (
        # (1e4, 10),
        # (1e5, 30),
        (1e6, 90),
        (1e7, 300),
        (1e8, 1800),
        (float("inf"), 3600),
    )
    try:
        api_config = APIConfig(api_config)
        first = None
        if api_config.args:
            first = api_config.args[0]
        elif api_config.kwargs:
            first = next(iter(api_config.kwargs.values()))
        if first is not None and hasattr(first, "shape"):
            total_elements = math.prod(first.shape)
            for threshold, timeout in TIMEOUT_STEPS:
                if total_elements <= threshold:
                    return timeout
    except Exception:
        pass
    return TIMEOUT_STEPS[-1][1]


def validate_gpu_options(options) -> tuple:
    """Validate and normalize GPU-related options."""
    if options.num_gpus == 0 and not options.gpu_ids.strip():
        return tuple()

    if options.num_gpus < -1:
        print(f"Invalid num_gpus: {options.num_gpus}, using all available.", flush=True)
        options.num_gpus = -1

    if options.gpu_ids:
        try:
            gpu_ids = [int(id) for id in options.gpu_ids.split(",") if id.strip()]
            gpu_ids = sorted(list(set(gpu_ids)))
            if not all(id >= 0 for id in gpu_ids) or -1 in gpu_ids:
                gpu_ids = [-1]
        except ValueError:
            print(
                f"Invalid gpu_ids: {options.gpu_ids}, using all available.", flush=True
            )
            gpu_ids = [-1]
    else:
        gpu_ids = [-1]

    if options.num_gpus > 0:
        if gpu_ids == [-1]:
            gpu_ids = list(range(options.num_gpus))
        elif len(gpu_ids) != options.num_gpus:
            print(
                f"num_gpus {options.num_gpus} mismatches gpu_ids length, using {len(gpu_ids)}.",
                flush=True,
            )
            options.num_gpus = len(gpu_ids)

    if options.num_workers_per_gpu <= 0 and options.num_workers_per_gpu != -1:
        print(
            f"Invalid num_workers_per_gpu: {options.num_workers_per_gpu}, using all available.",
            flush=True,
        )
        options.num_workers_per_gpu = -1

    if options.required_memory <= 0:
        print(
            f"Invalid required_memory: {options.required_memory}, setting to 10.0.",
            flush=True,
        )
        options.required_memory = 10.0

    return tuple(gpu_ids)


def check_gpu_memory(
    gpu_ids, num_workers_per_gpu, required_memory
):  # required_memory in GB
    assert isinstance(gpu_ids, tuple) and len(gpu_ids) > 0
    available_gpus = []
    max_workers_per_gpu = {}

    pynvml.nvmlInit()
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_ids = tuple(range(device_count)) if gpu_ids[0] == -1 else gpu_ids

        for gpu_id in gpu_ids:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                total_memory = int(mem_info.total) / (1024**3)  # Bytes to GB
                used_memory = int(mem_info.used) / (1024**3)  # Bytes to GB
                free_memory = total_memory - used_memory

                max_workers = int(free_memory // required_memory)
                if max_workers >= 1:
                    available_gpus.append(gpu_id)
                    max_workers_per_gpu[gpu_id] = (
                        max_workers
                        if num_workers_per_gpu == -1
                        else min(max_workers, num_workers_per_gpu)
                    )
            except pynvml.NVMLError as e:
                print(f"[WARNING] Failed to check GPU {gpu_id}: {str(e)}", flush=True)
                continue

    finally:
        pynvml.nvmlShutdown()

    return available_gpus, max_workers_per_gpu


def init_worker_gpu(
    gpu_worker_list, lock, available_gpus, max_workers_per_gpu, test_cpu
):
    set_engineV2()
    my_pid = os.getpid()

    def pid_exists(pid):
        try:
            os.kill(pid, 0)
            return True
        except OSError as e:
            return e.errno == errno.EPERM

    try:
        with lock:
            assigned_gpu = -1
            max_available_slots = -1
            for gpu_id in available_gpus:
                workers = gpu_worker_list[gpu_id]
                workers[:] = [pid for pid in workers if pid_exists(pid)]
                available_slots = max_workers_per_gpu[gpu_id] - len(workers)
                if available_slots > max_available_slots:
                    max_available_slots = available_slots
                    assigned_gpu = gpu_id

            if assigned_gpu == -1:
                raise RuntimeError(f"Worker {my_pid} could not be assigned a GPU.")

            gpu_worker_list[assigned_gpu].append(my_pid)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)

        import paddle
        import torch

        from tester import (APIConfig, APITestAccuracy, APITestCINNVSDygraph,
                            APITestPaddleOnly)

        globals()["torch"] = torch
        globals()["paddle"] = paddle
        globals()["APIConfig"] = APIConfig
        globals()["APITestAccuracy"] = APITestAccuracy
        globals()["APITestCINNVSDygraph"] = APITestCINNVSDygraph
        globals()["APITestPaddleOnly"] = APITestPaddleOnly

        def signal_handler(*args):
            torch.cuda.empty_cache()
            paddle.device.cuda.empty_cache()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        if test_cpu:
            paddle.device.set_device("cpu")

        print(
            f"{datetime.now()} Worker PID: {my_pid}, Assigned GPU ID: {assigned_gpu}",
            flush=True,
        )
    except Exception as e:
        print(
            f"{datetime.now()} Worker {my_pid} initialization failed: {e}", flush=True
        )
        raise


def run_test_case(api_config_str, options):
    """Run a single test case for the given API configuration."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_id = int(cuda_visible.split(",")[0])

    write_to_log("checkpoint", api_config_str)
    print(
        f"{datetime.now()} GPU {gpu_id} {os.getpid()} test begin: {api_config_str}",
        flush=True,
    )

    pynvml.nvmlInit()
    try:
        while True:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_memory = int(mem_info.free) / (1024**3)  # Bytes to GB

            if free_memory >= options.required_memory:
                break

            print(
                f"{datetime.now()} GPU {gpu_id} Free: {free_memory:.1f} GB, "
                f"Required: {options.required_memory:.1f} GB. ",
                "Waiting for available memory...",
                flush=True,
            )
            time.sleep(60)
    finally:
        pynvml.nvmlShutdown()

    try:
        api_config = APIConfig(api_config_str)
    except Exception as err:
        print(f"[config parse error] {api_config_str} {str(err)}", flush=True)
        return

    test_class = APITestAccuracy
    if options.paddle_only:
        test_class = APITestPaddleOnly
    elif options.paddle_cinn:
        test_class = APITestCINNVSDygraph
    elif options.accuracy:
        test_class = APITestAccuracy

    case = test_class(api_config, options.test_amp)
    try:
        case.test()
    except Exception as err:
        print(f"[test error] {api_config_str} {str(err)}", flush=True)
        if "CUDA error" in str(err) or "memory corruption" in str(err):
            exit(0)
    finally:
        case.clear_tensor()
        del case


def main():
    print(f"Main process id: {os.getpid()}")
    set_start_method("spawn")

    parser = argparse.ArgumentParser(description="API Test")
    parser.add_argument("--api_config_file", default="")
    parser.add_argument(
        "--api_config_file_pattern",
        default="",
        help="Pattern to match multiple config files (e.g., 'tester/api_config/api_config_support2torch_*.txt')",
    )
    parser.add_argument("--api_config", default="")
    parser.add_argument(
        "--paddle_only",
        default=False,
    )
    parser.add_argument(
        "--paddle_cinn",
        default=False,
    )
    parser.add_argument(
        "--accuracy",
        default=False,
    )
    parser.add_argument(
        "--test_amp",
        default=False,
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="Number of GPUs to use, -1 to use all available",
    )
    parser.add_argument(
        "--num_workers_per_gpu",
        type=int,
        default=1,
        help="Number of workers per GPU, -1 to maximize based on memory",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="",
        help="Comma-separated list of GPU IDs to use (e.g., 0,1,2), -1 for all available",
    )
    parser.add_argument(
        "--required_memory",
        type=float,
        default=10.0,
        help="Required memory per worker in GB",
    )
    parser.add_argument(
        "--test_cpu",
        type=bool,
        default=False,
        help="Whether to test CPU mode",
    )
    options = parser.parse_args()

    if options.api_config:
        # Single config execution
        from tester import (APIConfig, APITestAccuracy, APITestCINNVSDygraph,
                            APITestPaddleOnly)

        print(f"Test begin: {options.api_config}", flush=True)
        try:
            api_config = APIConfig(options.api_config)
        except Exception as err:
            print(f"[config parse error] {options.api_config} {str(err)}", flush=True)
            return

        test_class = APITestAccuracy
        if options.paddle_only:
            test_class = APITestPaddleOnly
        elif options.paddle_cinn:
            test_class = APITestCINNVSDygraph
        elif options.accuracy:
            test_class = APITestAccuracy

        case = test_class(api_config, options.test_amp)
        try:
            case.test()
        except Exception as err:
            print(f"[test error] {options.api_config}: {err}", flush=True)
        finally:
            case.clear_tensor()
            del case
    elif options.api_config_file or options.api_config_file_pattern:
        if options.api_config_file_pattern:
            import glob
            import re

            if "*" in options.api_config_file_pattern:
                all_files = glob.glob(options.api_config_file_pattern)
                regex_pattern = re.sub(r'\*([^*]*)$', r'\\d+\1', options.api_config_file_pattern)
                config_files = [
                    file for file in all_files if re.fullmatch(regex_pattern, file)
                ]
                if not config_files:
                    print(
                        f"No config files found: {options.api_config_file_pattern}",
                        flush=True,
                    )
                    return
            else:
                config_files = [options.api_config_file_pattern]
            print("Config files to be tested:", flush=True)
            for i, config_file in enumerate(sorted(config_files), 1):
                print(f"{i}. {config_file}", flush=True)
        else:
            config_files = [options.api_config_file]

        # Batch execution
        finish_configs = read_log("checkpoint")
        print(len(finish_configs), "cases have been tested.", flush=True)

        api_config_count = 0
        api_configs = set()
        for config_file in config_files:
            try:
                with open(config_file, "r") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    api_config_count += len(lines)
                    api_configs.update(lines)
            except FileNotFoundError:
                print(
                    f"Error: config file {config_file} not found",
                    flush=True,
                )
        print(api_config_count, "cases in total.", flush=True)
        dup_case = api_config_count - len(api_configs)
        if dup_case > 0:
            print(dup_case, "cases are duplicates and removed.", flush=True)

        api_config_count = len(api_configs)
        api_configs = sorted(api_configs - finish_configs)
        all_case = len(api_configs)
        fail_case = 0
        finish_case = api_config_count - all_case
        if finish_case:
            print(finish_case, "cases already tested.", flush=True)
        print(all_case, "cases will be tested.", flush=True)
        del api_config_count, dup_case, finish_case

        if options.num_gpus != 0 or options.gpu_ids:
            # Multi GPUs
            set_engineV2()
            BATCH_SIZE = 20000

            gpu_ids = validate_gpu_options(options)

            available_gpus, max_workers_per_gpu = check_gpu_memory(
                gpu_ids, options.num_workers_per_gpu, options.required_memory
            )
            if not available_gpus:
                print(
                    f"No GPUs with sufficient memory available. Current memory constraint is {options.required_memory} GB.",
                    flush=True,
                )
                return

            total_workers = sum(max_workers_per_gpu.values())
            print(
                f"Using {len(available_gpus)} GPU(s) with max workers per GPU: {max_workers_per_gpu}. Total workers: {total_workers}.",
                flush=True,
            )

            if options.test_cpu:
                print(f"Using {cpu_count()} CPU(s) for paddle in CPU mode.", flush=True)

            manager = Manager()
            gpu_worker_list = manager.dict(
                {gpu_id: manager.list() for gpu_id in available_gpus}
            )
            lock = Lock()

            pool = ProcessPool(
                max_workers=total_workers,
                initializer=init_worker_gpu,
                initargs=[
                    gpu_worker_list,
                    lock,
                    available_gpus,
                    max_workers_per_gpu,
                    options.test_cpu,
                ],
            )

            from tester import APIConfig

            def cleanup_handler(*args):
                cleanup(pool)
                sys.exit(0)

            signal.signal(signal.SIGINT, cleanup_handler)
            signal.signal(signal.SIGTERM, cleanup_handler)

            try:
                for batch_start in range(0, len(api_configs), BATCH_SIZE):
                    batch = api_configs[batch_start : batch_start + BATCH_SIZE]
                    futures = {}
                    for config in batch:
                        timeout = estimate_timeout(config)
                        future = pool.schedule(
                            run_test_case,
                            [config, options],
                            timeout=timeout,
                        )
                        futures[future] = config

                    for future in as_completed(futures):
                        config = futures[future]
                        try:
                            future.result()
                        except TimeoutError as exc:
                            write_to_log("timeout", config)
                            print(
                                f"[timeout] Test case timed out for {config}: {exc}",
                                flush=True,
                            )
                            fail_case += 1
                        except Exception as exc:
                            write_to_log("crash", config)
                            print(
                                f"[fatal] Worker crashed for {config}: {exc}",
                                flush=True,
                            )
                            fail_case += 1
                    aggregate_logs()
                print(f"{all_case} cases tested, {fail_case} failed.", flush=True)
                pool.close()
                pool.join()
            except Exception as e:
                print(f"Unexpected error: {e}", flush=True)
                cleanup(pool)
            finally:
                log_counts = aggregate_logs(end=True)
                print_log_info(all_case, log_counts)
        else:
            # Single worker
            from tester import (APIConfig, APITestAccuracy,
                                APITestCINNVSDygraph, APITestPaddleOnly)

            globals()["APIConfig"] = APIConfig
            globals()["APITestAccuracy"] = APITestAccuracy
            globals()["APITestCINNVSDygraph"] = APITestCINNVSDygraph
            globals()["APITestPaddleOnly"] = APITestPaddleOnly

            for config in api_configs:
                run_test_case(config, options)
            print(f"{all_case} cases tested.", flush=True)
    print("Done.")


if __name__ == "__main__":
    main()
