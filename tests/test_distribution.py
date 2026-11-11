import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zipfile

from ca2_forecast.products import archive_product_details, read_product_detail
from scripts.prepare_data import prepare_data


class DistributionTests(unittest.TestCase):
    def test_compressed_details_restore_and_preserve_existing_database(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            with sqlite3.connect(out / "detalls.sqlite") as db:
                db.execute("CREATE TABLE products (item_id TEXT PRIMARY KEY, detail TEXT)")
                db.execute("INSERT INTO products VALUES (?, ?)", ("A", json.dumps({"value": 3})))
            archive_product_details(out)
            (out / "detalls.sqlite").unlink()
            self.assertEqual(read_product_detail(out, "A"), {"value": 3})
            with sqlite3.connect(out / "detalls.sqlite") as db:
                db.execute("UPDATE products SET detail = ?", (json.dumps({"value": 4}),))
            self.assertEqual(read_product_detail(out, "A"), {"value": 4})

    def test_data_preparation_filters_exact_store_and_preserves_calendar(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "m5.zip"
            calendar = b"date,d\n2011-01-29,d_1\n"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("data/calendar.csv", calendar)
                for name in ("sales_train_validation.csv", "sales_train_evaluation.csv", "sell_prices.csv"):
                    zipped.writestr(name, "store_id,item_id,d_1\nCA_1,A,1\nCA_2,A,2\nCA_20,A,3\n")
                zipped.writestr("sample_submission.csv", "id,F1\nA_CA_2_validation,0\nA_CA_20_validation,0\nA_CA_2_evaluation,0\n")
            out = root / "data"
            counts = prepare_data(archive, out)
            self.assertEqual(counts["sales_train_validation.csv"], 1)
            self.assertEqual(counts["sample_submission.csv"], 2)
            self.assertEqual((out / "calendar.csv").read_bytes(), calendar)
            with self.assertRaises(FileExistsError):
                prepare_data(archive, out)


if __name__ == "__main__":
    unittest.main()
