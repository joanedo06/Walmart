"""Prepara les dades de CA_2 des del ZIP original de M5, sense sobreescriure."""

import argparse
import csv
import io
from pathlib import Path
import shutil
import tempfile
import zipfile

FILES = (
    "calendar.csv", "sales_train_validation.csv", "sales_train_evaluation.csv",
    "sell_prices.csv", "sample_submission.csv",
)
REQUIRED = FILES[:3]


def prepare_data(archive: Path, output_dir: Path) -> dict[str, int]:
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} ja existeix; indica una carpeta nova amb --output-dir")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with zipfile.ZipFile(archive) as zipped, tempfile.TemporaryDirectory(dir=output_dir.parent) as temporary:
        stage = Path(temporary) / "prepared"
        stage.mkdir()
        members = {}
        for name in FILES:
            matches = [entry for entry in zipped.infolist() if not entry.is_dir() and Path(entry.filename).name == name]
            if len(matches) > 1:
                raise ValueError(f"El ZIP conté més d'un {name}")
            if not matches and name in REQUIRED:
                raise ValueError(f"Falta {name} al ZIP de M5")
            if matches:
                members[name] = matches[0]
        for name, member in members.items():
            if name == "calendar.csv":
                with zipped.open(member) as source, (stage / name).open("wb") as target:
                    shutil.copyfileobj(source, target)
                continue
            with zipped.open(member) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as source, (stage / name).open("w", encoding="utf-8", newline="") as target:
                reader = csv.DictReader(source)
                if not reader.fieldnames:
                    raise ValueError(f"CSV buit: {name}")
                writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
                writer.writeheader()
                kept = 0
                for row in reader:
                    match = row["store_id"] == "CA_2" if "store_id" in row else row["id"].endswith(("_CA_2_validation", "_CA_2_evaluation"))
                    if match:
                        writer.writerow(row)
                        kept += 1
                if not kept:
                    raise ValueError(f"No hi ha files de CA_2 a {name}")
                counts[name] = kept
        stage.replace(output_dir)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="ZIP descarregat de M5 Forecasting Accuracy")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "m5-forecasting-accuracy")
    args = parser.parse_args()
    for filename, count in prepare_data(args.archive, args.output_dir).items():
        print(f"{filename}: {count:,} files de CA_2")


if __name__ == "__main__":
    main()
