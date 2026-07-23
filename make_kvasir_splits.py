import argparse
import hashlib
import json
import random
from pathlib import Path


def file_digest(paths):
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Create reproducible Kvasir-SEG splits.")
    parser.add_argument("--dataset", required=True, help="Kvasir-SEG directory")
    parser.add_argument("--output_dir", required=True, help="directory for split files")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument(
        "--two_way",
        action="store_true",
        help="Create train/evaluation split and mirror evaluation IDs to val.txt and test.txt",
    )

    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not 0 < args.train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not args.two_way:
        if not 0 <= args.val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        if args.train_ratio + args.val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be less than 1")

    dataset = Path(args.dataset).resolve()
    image_dir = dataset / "images"
    mask_dir = dataset / "masks"
    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file() and not path.name.startswith("."))
    image_names = [path.name for path in image_paths]
    missing_masks = [name for name in image_names if not (mask_dir / name).is_file()]
    extra_masks = sorted(
        path.name for path in mask_dir.iterdir()
        if path.is_file() and not path.name.startswith(".") and path.name not in set(image_names)
    )
    if missing_masks or extra_masks:
        raise RuntimeError(
            f"Dataset pairing failed: {len(missing_masks)} missing masks, {len(extra_masks)} extra masks"
        )

    shuffled = image_names.copy()
    random.Random(args.seed).shuffle(shuffled)
    train_end = int(len(shuffled) * args.train_ratio)
    if args.two_way:
        evaluation = sorted(shuffled[train_end:])
        splits = {
            "train": sorted(shuffled[:train_end]),
            "val": evaluation,
            "test": evaluation,
        }
    else:
        val_end = train_end + int(len(shuffled) * args.val_ratio)
        splits = {
            "train": sorted(shuffled[:train_end]),
            "val": sorted(shuffled[train_end:val_end]),
            "test": sorted(shuffled[val_end:]),
        }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        (output_dir / f"{name}.txt").write_text("\n".join(values) + "\n", encoding="utf-8")

    manifest = {
        "dataset": "Kvasir-SEG",
        "sample_count": len(image_names),
        "seed": args.seed,
        "split_mode": "train_eval_mirrored" if args.two_way else "train_val_test",
        "ratios": (
            {"train": args.train_ratio, "evaluation": round(1 - args.train_ratio, 10)}
            if args.two_way
            else {
                "train": args.train_ratio,
                "val": args.val_ratio,
                "test": round(1 - args.train_ratio - args.val_ratio, 10),
            }
        ),
        "counts": {name: len(values) for name, values in splits.items()},
        "overlap": {"val_test": len(set(splits["val"]) & set(splits["test"]))},
        "filename_sha256": file_digest(image_paths),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
