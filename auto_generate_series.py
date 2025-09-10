#!/usr/bin/env python3
"""
Automation runner for generate_pull_drawer_data_auto.py.

Runs a sequence of (base_episode, recover_episode) pairs. For each pair, it
invokes the generator in --all_record mode to record two episodes per
iteration (pull then recover) and repeats for --num_episodes iterations.

Resilience:
- Monitors stdout for "[ERROR]" and restarts the generator if seen.
- Restarts the generator if no output is produced for --stall_timeout seconds.

Usage example:
  python3 auto_generate_series.py \
    --pairs 0:0 1:1 2:2 \
    --chunks 3 --episodes_per_chunk 5 \
    --base_dataset_dir datasets/pull \
    --recover_dataset_dir datasets/push \
    --new_dataset_dir datasets/out \
    --gen_script generate_pull_drawer_data_auto.py

Notes:
- Additional args can be passed through to the generator after "--", e.g.:
    ... -- --pos_x_range -0.1 0.1 --pos_y_range -0.3 0.0 --debug
"""

import argparse
import os
import signal
import sys
import threading
import time
from subprocess import Popen, PIPE, STDOUT
from queue import Queue, Empty


def _enqueue_output(stream, queue, on_line=None):
    for line in iter(stream.readline, ''):
        if on_line:
            try:
                on_line(line)
            except Exception:
                pass
        queue.put(line)
    stream.close()


def run_generator_once(python_bin, gen_script, gen_args, stall_timeout, expected_num_episodes, print_prefix="GEN"):
    """Run the generator once with monitoring.

    Returns True if completed without detected error/timeout, False otherwise.
    """
    cmd = [python_bin, gen_script] + gen_args
    print(f"[{print_prefix}] Launching: {' '.join(cmd)}")

    last_output_time = time.time()
    saw_error = False
    saw_generation_complete = False
    success_final_ok = False
    episode_success_count = 0

    import re
    re_success_final = re.compile(r"Successfully generated\s+(\d+)\/(\d+)\s+episodes")
    re_episode_ok = re.compile(r"Episode\s+(\d+)\s+completed successfully")

    def on_line(line):
        nonlocal last_output_time, saw_error
        last_output_time = time.time()
        # Simple error detector: any line containing [ERROR]
        if "[ERROR]" in line:
            saw_error = True
        # Success by final summary
        m = re_success_final.search(line)
        if m:
            try:
                got, total = int(m.group(1)), int(m.group(2))
                if got == total == expected_num_episodes:
                    success_final_ok = True
            except Exception:
                pass
        # Track per-episode success lines as a fallback
        m2 = re_episode_ok.search(line)
        if m2:
            try:
                episode_success_count = max(episode_success_count, int(m2.group(1)))
            except Exception:
                pass
        if "=== Generation Complete ===" in line:
            saw_generation_complete = True

    # Start process in its own process group for clean SIGINT
    proc = Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True, bufsize=1, universal_newlines=True,
                preexec_fn=os.setsid)
    q = Queue()
    t = threading.Thread(target=_enqueue_output, args=(proc.stdout, q, on_line), daemon=True)
    t.start()

    try:
        while True:
            # Drain and print any available output lines without blocking long
            try:
                while True:
                    line = q.get_nowait()
                    print(line, end='')
            except Empty:
                pass

            # Check process status
            ret = proc.poll()
            if ret is not None:
                # Process finished; flush remaining output
                try:
                    while True:
                        line = q.get_nowait()
                        print(line, end='')
                except Empty:
                    pass
                # Prefer success signaled by logs
                if success_final_ok or (saw_generation_complete and episode_success_count >= expected_num_episodes and not saw_error):
                    print(f"[{print_prefix}] Completed successfully (episodes={episode_success_count}).")
                    return True
                print(f"[{print_prefix}] Exited with code {ret} (error_seen={saw_error}, episodes={episode_success_count}, final_ok={success_final_ok}).")
                return False

            # Timeout monitoring
            if time.time() - last_output_time > stall_timeout:
                print(f"[{print_prefix}] Stall detected (> {stall_timeout}s without output). Restarting...")
                try:
                    os.killpg(proc.pid, signal.SIGINT)
                except Exception:
                    pass
                time.sleep(2.0)
                try:
                    proc.terminate()
                except Exception:
                    pass
                return False

            # Error monitoring
            if saw_error:
                print(f"[{print_prefix}] [ERROR] detected in output. Restarting...")
                try:
                    os.killpg(proc.pid, signal.SIGINT)
                except Exception:
                    pass
                time.sleep(2.0)
                try:
                    proc.terminate()
                except Exception:
                    pass
                return False

            time.sleep(0.25)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass


def parse_pairs(pairs_list):
    result = []
    for token in pairs_list:
        if ':' not in token:
            raise ValueError(f"Invalid pair '{token}', expected format base:recover (e.g., 3:5)")
        b, r = token.split(':', 1)
        result.append((int(b), int(r)))
    return result


def main():
    ap = argparse.ArgumentParser(description="Automation runner for generate_pull_drawer_data_auto.py")
    ap.add_argument('--pairs', nargs='+', required=True, help="List of base:recover episode index pairs (e.g., 0:0 1:1 2:2)")
    ap.add_argument('--chunks', type=int, default=3, help="Number of generator invocations per pair (default: 3)")
    ap.add_argument('--episodes_per_chunk', type=int, default=5, help="--num_episodes to pass to generator for each chunk (default: 5)")
    ap.add_argument('--num_episodes', type=int, default=None, help="Deprecated. If set, runs a single chunk with this many iterations.")
    ap.add_argument('--base_dataset_dir', type=str, required=True, help="Path to pull (base) dataset directory")
    ap.add_argument('--recover_dataset_dir', type=str, required=True, help="Path to recover (push) dataset directory")
    ap.add_argument('--new_dataset_dir', type=str, required=True, help="Path to output dataset directory")
    ap.add_argument('--gen_script', type=str, default='generate_pull_drawer_data_auto.py', help="Path to the generator script")
    ap.add_argument('--python', type=str, default=sys.executable, help="Python interpreter to use")
    ap.add_argument('--stall_timeout', type=float, default=30.0, help="Seconds of no output before restart")
    ap.add_argument('--max_retries', type=int, default=5, help="Max restarts per pair before giving up")
    ap.add_argument('--recover_start_frame', type=int, default=0, help="Start frame for recover episodes (passed to generator)")
    # Toggle all_record vs. single-record mode for the generator
    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument('--all_record', dest='all_record', action='store_true', help='Run generator in --all_record mode (default)')
    mode_group.add_argument('--no_all_record', dest='all_record', action='store_false', help='Run generator without --all_record')
    ap.set_defaults(all_record=True)

    ap.add_argument('--gen_extra', nargs=argparse.REMAINDER, help="Extra args to pass through to the generator after --")

    args = ap.parse_args()

    pairs = parse_pairs(args.pairs)

    for idx, (base_idx, recover_idx) in enumerate(pairs, start=1):
        print(f"\n=== Pair {idx}/{len(pairs)}: base={base_idx}, recover={recover_idx} ===")

        # Determine chunking configuration
        chunks = args.chunks
        episodes_per_chunk = args.episodes_per_chunk
        if args.num_episodes is not None:
            chunks = 1
            episodes_per_chunk = int(args.num_episodes)

        for c in range(1, chunks + 1):
            print(f"-- Pair {idx}: Chunk {c}/{chunks} (num_episodes={episodes_per_chunk}) --")

            # Build generator args for this chunk
            gen_args = [
                '--num_episodes', str(episodes_per_chunk),
                '--base_episode', str(base_idx),
                '--recover_episode', str(recover_idx),
                '--recover_start_frame', str(args.recover_start_frame),
                '--base_dataset_dir', args.base_dataset_dir,
                '--recover_dataset_dir', args.recover_dataset_dir,
                '--new_dataset_dir', args.new_dataset_dir,
            ]
            if args.all_record:
                gen_args.insert(0, '--all_record')

            if args.gen_extra:
                # Strip leading '--' if present because argparse.REMAINDER may keep it
                extra = args.gen_extra
                if len(extra) > 0 and extra[0] == '--':
                    extra = extra[1:]
                gen_args += extra

            attempts = 0
            ok = False
            while attempts <= args.max_retries:
                attempts += 1
                ok = run_generator_once(
                    args.python, args.gen_script, gen_args, args.stall_timeout,
                    expected_num_episodes=episodes_per_chunk,
                    print_prefix=f"PAIR {idx} CHUNK {c} ATTEMPT {attempts}"
                )
                if ok:
                    print(f"[PAIR {idx} CHUNK {c}] Finished successfully.")
                    break
                else:
                    print(f"[PAIR {idx} CHUNK {c}] Attempt {attempts} failed. Restarting...")
                    time.sleep(2.0)

            if attempts > args.max_retries and not ok:
                print(f"[PAIR {idx} CHUNK {c}] Reached max retries ({args.max_retries}). Moving on to next chunk/pair.")

    print("\nAll pairs processed.")


if __name__ == '__main__':
    main()
