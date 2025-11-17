"""
OpenVLA control loop using franky (no ROS). Mirrors the behavior of query_openvla.py.
"""

import argparse

from franky_vla_runner import FrankyVLARunner, parse_common_args, prepare_record_dir, save_run_metadata


def parse_arguments():
    parser = parse_common_args(default_record_dir="logs/openvla-franky")
    return parser.parse_args()


def main():
    args = parse_arguments()
    args.record_dir = prepare_record_dir(args.record_dir, prefix="OpenVLA")
    save_run_metadata(args.record_dir, args)

    runner = FrankyVLARunner(args, include_state=False)
    runner.run()


if __name__ == "__main__":
    main()

