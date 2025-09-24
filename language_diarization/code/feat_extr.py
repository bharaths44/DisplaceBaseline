#
#
import textwrap
import argparse
import logging

import os
import pandas as pd
import numpy as np
import whisper


def get_args():
    # audio_file_path = '/home1/pratik/lang_diarization/data/dev'
    # output_dir = '/home1/pratik/lang_diarization/output/whisper_dev_10_0.4'
    # segment_folder = '/home1/pratik/lang_diarization/output/dev_subseg'
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""
        Creates subsegments files from input segments files in a folder.

        ... (rest of the description) ...
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--aud_path",
        type=str,
        default=None,
        help=" Directory containing the audio files",
    )
    parser.add_argument(
        "--seg_dir",
        type=str,
        default=None,
        help=" Directory where the windowed segments are available",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help=" Directory where features are written to",
    )

    args = parser.parse_args()
    return args


S_RATE = 16000  # Samples/second


def whisper_call(wav, segments):
    print("wavfile = ", wav)
    model = whisper.load_model("base")
    audio = whisper.load_audio(wav)
    audio_len = audio.shape[0]
    output = pd.DataFrame()

    for _, speaker_id, start, end in segments:
        start_sample = int(float(start) * S_RATE)
        end_sample = int(float(end) * S_RATE)

        audio_seg = audio[start_sample:end_sample]
        audio_seg = whisper.pad_or_trim(audio_seg)
        mel_seg = whisper.log_mel_spectrogram(audio_seg).to(model.device)
        _, prob_seg = model.detect_language(mel_seg)
        # print(f"Speaker {speaker_id}: Detected language: {max(prob_seg, key=prob_seg.get)}")
        df_frame = pd.DataFrame([prob_seg])
        output = pd.concat([output, df_frame], ignore_index=True)

    return output


def save_embeddings(audio_folder_path, output_dir, segment_folder):
    for filename in os.listdir(audio_folder_path):
        try:
            if filename.endswith(".wav"):
                audio_file_path = os.path.join(audio_folder_path, filename)
                segment_file_name = (
                    f"subsegments_{filename.replace('.wav', 'pyannote.segment.txt')}"
                )
                segment_file_path = os.path.join(segment_folder, segment_file_name)

                # Read segments from file
                with open(segment_file_path, "r") as f:
                    segments = [line.strip().split() for line in f]

                pred = whisper_call(audio_file_path, segments)
                # print(pred)

                # Ensure the output directory exists
                os.makedirs(output_dir, exist_ok=True)

                # Save the numpy array
                output_filename = os.path.splitext(filename)[0] + "_output.npy"
                np.save(os.path.join(output_dir, output_filename), pred.to_numpy())
        except Exception as e:
            print(f"Error processing {filename}: {e}")


############################### MAIN #############################################################

# audio_file_path = '/home1/pratik/lang_diarization/data/dev'
# output_dir = '/home1/pratik/lang_diarization/output/whisper_dev_10_0.4'
# segment_folder = '/home1/pratik/lang_diarization/output/dev_subseg'
# save_embeddings(audio_file_path, output_dir, segment_folder)


def main():
    args = get_args()
    try:
        save_embeddings(args.aud_path, args.out_dir, args.seg_dir)
    except Exception:
        logging.error("Failed creating subsegments", exc_info=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
