#! /usr/bin/env python

# Copyright 2017  Vimal Manohar
#           2017  Matthew Maciejewski

# Modified by Shreyas Ramoji
# Apache 2.0.

from __future__ import print_function
import argparse
import logging
import sys
import textwrap
from pdb import set_trace as bp
import os


def get_args():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""
        Creates subsegments files from input segments files in a folder.

        ... (rest of the description) ...
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--max-segment-duration",
        type=float,
        default=30,
        help="""Maximum duration of the
                        subsegments (in seconds)""",
    )
    parser.add_argument(
        "--overlap-duration",
        type=float,
        default=5,
        help="""Overlap between
                        adjacent segments (in seconds)""",
    )
    parser.add_argument(
        "--max-remaining-duration",
        type=float,
        default=10,
        help="""Segment is not split
                        if the left-over duration is more than this
                        many seconds""",
    )
    parser.add_argument(
        "--constant-duration",
        type=bool,
        default=False,
        help="""Final segment is given
                        a start time max-segment-duration before the
                        end to force a constant segment duration. This
                        overrides the max-remaining-duration parameter""",
    )
    parser.add_argument(
        "--input_folder", type=str, help="""Input folder containing segment files"""
    )
    parser.add_argument(
        "--out_subsegments_folder",
        type=str,
        help="""Output folder for subsegments files""",
    )

    args = parser.parse_args()
    return args


def process_segment_file(
    segment_file,
    out_subsegments_file,
    max_segment_duration,
    overlap_duration,
    max_remaining_duration,
    constant_duration,
):
    if constant_duration:
        dur_threshold = max_segment_duration
    else:
        dur_threshold = max_segment_duration + max_remaining_duration

    with open(segment_file, "r") as f:
        vad_segments = f.readlines()

    with open(out_subsegments_file, "w") as of:
        for line in vad_segments:
            parts = line.strip().split()
            utt_id = parts[0]
            utt_id0 = parts[1]
            start_time = float(parts[2])
            end_time = float(parts[3])

            dur = end_time - start_time

            start = start_time
            while dur > dur_threshold:
                end = start + max_segment_duration
                start_relative = start - start_time
                end_relative = end - start_time
                new_utt = "{utt_id}-{s:08d}-{e:08d}".format(
                    utt_id=utt_id,
                    s=int(100 * start_relative),
                    e=int(100 * end_relative),
                )
                of.write(
                    "{new_utt} {utt_id0} {s:.3f} {e:.3f}\n".format(
                        new_utt=new_utt,
                        utt_id0=utt_id0,
                        s=start_relative + start_time,
                        e=start_relative + max_segment_duration + start_time,
                    )
                )
                start += max_segment_duration - overlap_duration
                dur -= max_segment_duration - overlap_duration

            if constant_duration:
                if dur < 0:
                    continue
                if dur < max_remaining_duration:
                    start = max(end_time - max_segment_duration, start_time)
                end = min(start + max_segment_duration, end_time)
            else:
                end = end_time
            new_utt = "{utt_id}-{s:08d}-{e:08d}".format(
                utt_id=utt_id,
                s=int(round(100 * (start - start_time))),
                e=int(round(100 * (end - start_time))),
            )
            if out_subsegments_file is not None:
                of.write(
                    "{new_utt} {utt_id0} {s:.3f} {e:.3f}\n".format(
                        new_utt=new_utt, utt_id0=utt_id0, s=start, e=end
                    )
                )


def run(
    input_folder,
    out_subsegments_folder,
    max_segment_duration,
    overlap_duration,
    max_remaining_duration,
    constant_duration,
):
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_segment_file = os.path.join(input_folder, filename)
            output_subsegments_file = os.path.join(
                out_subsegments_folder, f"subsegments_{filename}"
            )
            os.makedirs(out_subsegments_folder, exist_ok=True)

            process_segment_file(
                input_segment_file,
                output_subsegments_file,
                max_segment_duration,
                overlap_duration,
                max_remaining_duration,
                constant_duration,
            )


def main():
    args = get_args()
    try:
        run(**vars(args))
    except Exception:
        logging.error("Failed creating subsegments", exc_info=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
