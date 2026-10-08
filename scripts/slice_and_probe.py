"""
Fused alignment, slicing and statistics in one GPU pass. No wav files written.

Why fused
    Slicing the eval partition to disk would write on the order of 145,000 tiny
    wav files. That is slow on Colab local disk, unusable over a Drive mount,
    and a session timeout loses all of it. The probe only ever consumes
    per-slice statistics, so the wavs are pure overhead. This aligns, slices in
    memory, reduces each slice to nine scalars and appends one row to a table.
    Output is a few megabytes.

Both segmentations, one pass
    Each phoneme run yields two rows, seg=run for the true extent and
    seg=window for a fixed 60ms window centred on the run's first frame, which
    reproduces the older behaviour. Same audio, same alignment, same model, so
    the segmentation comparison is exactly controlled. That matters because the
    pilot effect was measured on fixed windows, and without this a null result
    would be confounded between sample size and segmentation.

Resumable
    Rows are appended and flushed every --checkpoint-every utterances, and
    already processed utterance ids are skipped on restart. A reclaimed Colab
    session costs you the last few utterances, not the run.

Speaker column
    Every row carries a speaker id, so that bootstrap resampling can
    eventually be done per speaker rather than per utterance, and so an
    auditor can check speaker level clustering at all. For --subset and
    --hf-dataset, it is read from column 0 of the ASVspoof 2019 LA protocol
    file, keyed on the utterance id in column 1. An utterance id absent from
    the protocol writes UNKNOWN rather than a blank cell, which would
    otherwise be silently indistinguishable from a genuinely empty field.
    --manifest corpora carry no ASVspoof protocol, so their rows write "-",
    the same not-applicable sentinel already used for attack, language and
    corpus when a manifest does not supply them. This column is not yet
    consumed by any analysis script; wiring it into clustering is separate
    work.

Two input modes
    --subset reads the ASVspoof 2019 LA protocols directly. Use this for the
    pre-registered test.

    --manifest reads any corpus from a CSV, so the same measurement runs on
    MLAAD, In-the-Wild, a Bengali corpus or anything else. This is what H2, the
    cross lingual test, needs, and it is what makes the tool useful to anyone
    who is not working on ASVspoof.

    Manifest columns. path is required. label is required and accepts bonafide
    and spoof or 0 and 1. utterance_id, attack, language and corpus are
    optional, and utterance_id defaults to the filename stem. Carrying language
    through matters, because the cross lingual analysis groups on it.

        path,label,utterance_id,language
        data/mlaad/de/real/0001.wav,bonafide,de_0001,de
        data/mlaad/de/fake/0001.wav,spoof,de_f0001,de

    The alignment model is multilingual and emits IPA, so the phonetic space is
    shared across languages by construction. That is the whole reason a cross
    lingual comparison is meaningful here rather than apples to oranges.

Usage
    python scripts/slice_and_probe.py --subset eval \\
        --limit-bonafide 1500 --limit-spoof 1500 --out results/eval_slices.csv

    python scripts/slice_and_probe.py --manifest manifests/mlaad_de.csv \\
        --out results/mlaad_de_slices.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time


# Quiet the weight loading bar before transformers is imported anywhere. It
# emits several hundred lines per model load, which makes a Colab log unreadable
# and buries the per utterance progress that is actually informative.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import librosa
import soundfile as sf
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from phonophase.config import load_config  # noqa: E402
from phonophase.corpus import partition_paths, read_protocol, select_balanced  # noqa: E402
from phonophase.probe import STATISTIC_NAMES, slice_statistics  # noqa: E402

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
SR = 16000
FRAME_SEC = 0.02      # wav2vec2 emits one frame per 20ms
MARGIN_SEC = 0.02
MIN_SLICE_SEC = 0.03
WINDOW_SEC = 0.06     # the legacy fixed window, kept for the comparison

PHONEME_MAP = {
    "PLOSIVES": ["p", "t", "k", "b", "d", "g"],
    "FRICATIVES": ["s", "z", "f", "v", "ʃ", "ʒ", "h", "θ", "ð"],
    "AFFRICATES": ["tʃ", "dʒ"],
    "NASALS": ["m", "n", "ŋ"],
    "APPROXIMANTS": ["l", "r", "w", "j"],
    "VOWELS_STEADY": ["a", "e", "i", "o", "u", "æ", "ʌ", "ɔ", "ə", "ɪ", "ʊ", "ɛ", "ɑ"],
    "VOWELS_MOVING": ["aɪ", "eɪ", "ɔɪ", "aʊ", "oʊ"],
}
CHAR_TO_CATEGORY = {c: cat for cat, chars in PHONEME_MAP.items() for c in chars}

UNKNOWN_SPEAKER = "UNKNOWN"

FIELDS = ["utterance_id", "speaker", "label", "attack", "language", "corpus",
          "category", "phoneme", "seg", "start_frame", "n_frames",
          "duration_s"] + STATISTIC_NAMES


def read_speaker_map(protocol_path):
    """
    {utterance_id: speaker_id}, read directly from an ASVspoof 2019 LA
    protocol file.

    Lines are whitespace separated: SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID
    KEY. Column 0 is the speaker, column 1 the utterance id every slice table
    is keyed on. phonophase.corpus.read_protocol already parses this same
    file but keeps only columns 1, 3 and 4 (utterance id, attack, label), so
    the speaker column is read again here rather than by widening that
    shared function, which this task's scope does not touch.
    """
    mapping = {}
    with open(protocol_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                mapping[parts[1]] = parts[0]
    return mapping


def speaker_for(utt, speaker_map):
    """
    Resolve one utterance id to its speaker id.

    A miss means the protocol file does not agree with the utterance ids
    actually being processed. That is a data problem worth seeing, so it
    surfaces as an explicit sentinel that can never collide with a real
    ASVspoof speaker id (all of which look like LA_####), rather than "",
    which would be indistinguishable from a genuinely empty field once it is
    sitting in a CSV cell.
    """
    return speaker_map.get(utt, UNKNOWN_SPEAKER)


def check_header_compatible(out_path, fields):
    """
    Refuse to resume into a file whose header predates this column layout.

    Resuming appends rows shaped like `fields` under whatever header line is
    already on disk. Adding a column, such as this one adding speaker, is
    exactly the change that would silently misalign every row appended after
    the old header. Caught here, before the GPU model loads, rather than
    discovered later by a reader that gets the wrong values in the wrong
    columns.
    """
    if not os.path.exists(out_path):
        return
    with open(out_path, newline="") as fh:
        header = next(csv.reader(fh), [])
    if header and header != fields:
        raise SystemExit(
            f"{out_path} already exists with a header that does not match "
            "the current columns (this version adds 'speaker'). Appending "
            "would misalign every row written after it. Start a new --out "
            "file, or migrate the existing file to the new column layout "
            "before resuming."
        )


def read_manifest(path):
    """
    Load an arbitrary corpus. Returns (audio_path, utterance_id, attack,
    language, corpus, label_str, speaker) per row.

    A relative path is resolved against the manifest's own directory first, so
    an exported subset works wherever it is unzipped without anyone having to
    match the working directory to the archive layout.

    speaker is always "-". A manifest describes an arbitrary corpus with no
    ASVspoof protocol to source a speaker id from, so there is nothing
    truthful to fill in here; "-" is the same not-applicable sentinel already
    used above for attack, language and corpus when a manifest omits them.
    """
    manifest_dir = os.path.dirname(os.path.abspath(path))
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            audio = r.get("path", "").strip()
            if not audio:
                continue
            label = str(r.get("label", "")).strip().lower()
            if label in ("0", "bonafide", "real", "genuine"):
                label = "bonafide"
            elif label in ("1", "spoof", "fake", "synthetic"):
                label = "spoof"
            else:
                continue
            if not os.path.isabs(audio) and not os.path.exists(audio):
                candidate = os.path.join(manifest_dir, audio)
                if os.path.exists(candidate):
                    audio = candidate

            rows.append((
                audio,
                r.get("utterance_id") or os.path.splitext(os.path.basename(audio))[0],
                r.get("attack") or "-",
                r.get("language") or "-",
                r.get("corpus") or "-",
                label,
                "-",
            ))
    return rows


def group_runs(chars):
    """
    Consecutive identical predictions become one run, as (char, first, last).

    Caveat worth stating. CTC argmax is peaky, so a phoneme is usually emitted
    for one or two frames with blanks around it rather than spanning its true
    acoustic duration. These runs are therefore better than the old fixed
    window, and observed durations do vary rather than being constant, but they
    are still not forced alignment boundaries. Proper boundaries would need
    Viterbi forced alignment against a transcript, which this corpus does not
    ship. Treat seg=run as an improved approximation, not ground truth.
    """
    runs, start = [], 0
    for i in range(1, len(chars) + 1):
        if i == len(chars) or chars[i] != chars[start]:
            runs.append((chars[start], start, i))
            start = i
    return runs


def already_done(path):
    if not os.path.exists(path):
        return set()
    seen = set()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            seen.add(row["utterance_id"])
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--subset", default=None, choices=["train", "dev", "eval"],
                    help="ASVspoof 2019 LA partition. Use this for the "
                         "pre-registered test")
    ap.add_argument("--manifest", default=None,
                    help="CSV describing any other corpus, see the module "
                         "docstring. Use this for the cross lingual test")
    ap.add_argument("--hf-dataset", default=None,
                    help="HuggingFace dataset id holding the corpus, for "
                         "example SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA. "
                         "Avoids the host rate limiting that makes the direct "
                         "download fail from Colab")
    ap.add_argument("--hf-split", default="test")
    ap.add_argument("--protocol", default=None,
                    help="protocol file supplying attack ids, which the HF "
                         "mirrors do not carry. Defaults to the copy shipped in "
                         "configs/protocols/")
    ap.add_argument("--limit-bonafide", type=int, default=1500)
    ap.add_argument("--limit-spoof", type=int, default=1500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--segmentations", default="run,window",
                    help="comma separated, any of run and window")
    args = ap.parse_args()

    segs = [s.strip() for s in args.segmentations.split(",") if s.strip()]

    modes = [bool(args.subset), bool(args.manifest), bool(args.hf_dataset)]
    if sum(modes) != 1:
        raise SystemExit("pass exactly one of --subset, --manifest or --hf-dataset")

    if args.hf_dataset:
        run_from_hf(args, segs)
        return

    if args.manifest:
        items = read_manifest(args.manifest)
        done = already_done(args.out)
        if done:
            print(f"resuming, {len(done)} utterances already in {args.out}")
        items = [r for r in items if r[1] not in done]
        n_b = sum(1 for r in items if r[5] == "bonafide")
        print(f"manifest {args.manifest}: {len(items)} to process, "
              f"{n_b} bonafide, {len(items) - n_b} spoof")
        languages = sorted({r[3] for r in items})
        print(f"languages: {', '.join(languages)}")
        run(items, args, segs)
        return

    config = load_config(args.config)
    audio_dir, protocol = partition_paths(config, args.subset)
    for path in (audio_dir, protocol):
        if not os.path.exists(path):
            raise SystemExit(
                f"missing: {path}\n\n"
                "On Colab the corpus download is often rate limited by the host, "
                "which returns 403. Run scripts/export_audio_subset.py locally "
                "instead and use the --manifest route, which needs about 190 MB "
                "rather than 7.12 GB."
            )

    speaker_map = read_speaker_map(protocol)
    selected = select_balanced(read_protocol(protocol), args.limit_bonafide,
                               args.limit_spoof, args.seed)
    done = already_done(args.out)
    if done:
        print(f"resuming, {len(done)} utterances already in {args.out}")
    selected = [r for r in selected if r[0] not in done]

    n_bona = sum(1 for r in selected if r[2] == "bonafide")
    print(f"{args.subset}: {len(selected)} to process, {n_bona} bonafide, "
          f"{len(selected) - n_bona} spoof")

    items = [(os.path.join(audio_dir, f"{utt}.flac"), utt, attack, "en",
              "asvspoof2019_la", label, speaker_for(utt, speaker_map))
             for utt, attack, label in selected]
    run(items, args, segs)


DEFAULT_EVAL_PROTOCOL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "configs", "protocols",
    "ASVspoof2019.LA.cm.eval.trl.txt")


def run_from_hf(args, segs):
    """
    Stream the corpus from a HuggingFace mirror instead of the original host.

    Why this exists. Edinburgh DataShare rate limits by IP and returns 403 to
    Colab, whose egress addresses are shared and usually already over quota.
    The mirrors are redistributed under ODC-BY, which permits it, so this is a
    licensing-clean way to make the whole run self contained.

    The mirrors carry path, audio and label but NOT the attack id, so the
    protocol file supplies that. It is 2.2 MB and ships in configs/protocols/,
    which also keeps the selection identical to every other route since it
    feeds the same select_balanced.
    """
    from datasets import load_dataset

    protocol = args.protocol or DEFAULT_EVAL_PROTOCOL
    if not os.path.exists(protocol):
        raise SystemExit(f"protocol not found: {protocol}")

    speaker_map = read_speaker_map(protocol)
    selected = select_balanced(read_protocol(protocol), args.limit_bonafide,
                               args.limit_spoof, args.seed)
    wanted = {utt: (attack, label) for utt, attack, label in selected}

    done = already_done(args.out)
    if done:
        print(f"resuming, {len(done)} utterances already in {args.out}")
    remaining = {k: v for k, v in wanted.items() if k not in done}

    n_bona = sum(1 for a, l in remaining.values() if l == "bonafide")
    print(f"{args.hf_dataset}: {len(remaining)} to process, {n_bona} bonafide, "
          f"{len(remaining) - n_bona} spoof")
    print("streaming, so only the shards containing wanted rows are pulled")

    ds = load_dataset(args.hf_dataset, split=args.hf_split, streaming=True)

    def items():
        for row in ds:
            utt = os.path.splitext(os.path.basename(row["path"]))[0]
            if utt not in remaining:
                continue
            attack, label = remaining[utt]
            # Audio arrives decoded, so nothing is written to disk at any point.
            yield (row["audio"], utt, attack, "en", args.hf_dataset, label,
                   speaker_for(utt, speaker_map))

    run(items(), args, segs, total=len(remaining), audio_is_array=True)


def run(items, args, segs, total=None, audio_is_array=False):
    """Align, slice and reduce. Shared by both input modes."""
    check_header_compatible(args.out, FIELDS)
    print(f"segmentations: {', '.join(segs)}")

    from transformers import (Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC,
                              Wav2Vec2PhonemeCTCTokenizer)
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cpu":
        print("WARNING: this is a GPU job. On CPU it will take many hours.")

    # Loaded as feature extractor plus tokenizer rather than through
    # Wav2Vec2Processor, and with do_phonemize disabled.
    #
    # Wav2Vec2Processor builds the phoneme tokenizer with phonemization on,
    # which initialises an espeak backend at load time and raises
    # "espeak not installed on your system" on any machine without it,
    # including a stock Colab runtime. We never phonemize text. We only decode
    # the model's CTC output into IPA symbols, which needs the vocabulary and
    # nothing else. Disabling it removes a system dependency outright.
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(MODEL_ID, do_phonemize=False)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device).eval()
    inv_vocab = {v: k for k, v in tokenizer.get_vocab().items()}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    new_file = not os.path.exists(args.out)
    fh = open(args.out, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()

    started = time.time()
    written = 0
    for n, (audio_src, utt, attack, language, corpus, label_str, speaker) in enumerate(
            tqdm(items, desc="utterances", total=total), 1):
        try:
            if audio_is_array:
                speech, sr = np.asarray(audio_src["array"]), audio_src["sampling_rate"]
            else:
                speech, sr = sf.read(audio_src)
            if speech.ndim > 1:
                speech = speech[:, 0]
            if sr != SR:
                # Other corpora are not all 16 kHz, and the alignment model
                # requires it. Resample rather than silently dropping them.
                speech = librosa.resample(speech.astype(np.float64),
                                          orig_sr=sr, target_sr=SR)
            speech = speech.astype(np.float64)

            inputs = feature_extractor(speech, sampling_rate=SR,
                                       return_tensors="pt").input_values.to(device)
            with torch.no_grad():
                logits = model(inputs).logits
            chars = [inv_vocab[t.item()].lower() for t in torch.argmax(logits, dim=-1)[0]]

            label = 0 if label_str == "bonafide" else 1
            if speaker == UNKNOWN_SPEAKER:
                print(f"  no speaker id for {utt} (not in the protocol file)")

            for char, first, last in group_runs(chars):
                category = CHAR_TO_CATEGORY.get(char)
                if not category:
                    continue

                spans = {}
                if "run" in segs:
                    spans["run"] = (first * FRAME_SEC - MARGIN_SEC,
                                    last * FRAME_SEC + MARGIN_SEC)
                if "window" in segs:
                    # The legacy fixed window, centred on the run's first frame.
                    centre = (first + 0.5) * FRAME_SEC
                    spans["window"] = (centre - WINDOW_SEC / 2, centre + WINDOW_SEC / 2)

                for seg, (s0, s1) in spans.items():
                    a = max(0, int(s0 * SR))
                    b = min(len(speech), int(s1 * SR))
                    if (b - a) <= MIN_SLICE_SEC * SR:
                        continue
                    stats = slice_statistics(speech[a:b])
                    if stats is None:
                        continue
                    writer.writerow({
                        "utterance_id": utt, "speaker": speaker, "label": label,
                        "attack": attack, "language": language, "corpus": corpus,
                        "category": category, "phoneme": char, "seg": seg,
                        "start_frame": first, "n_frames": last - first,
                        "duration_s": round((b - a) / SR, 4), **stats,
                    })
                    written += 1
        except Exception as exc:
            print(f"  skipped {utt}: {exc}")

        if n % args.checkpoint_every == 0:
            fh.flush()
            os.fsync(fh.fileno())

    fh.close()
    elapsed = time.time() - started
    print(f"\nwrote {written} rows to {args.out} in {elapsed/60:.1f} min")
    print(f"file size {os.path.getsize(args.out)/1e6:.1f} MB")
    print("\nDownload that one file. The analysis runs locally in seconds.")


if __name__ == "__main__":
    main()
