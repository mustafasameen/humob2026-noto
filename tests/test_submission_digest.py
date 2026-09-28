"""The rounded submission digest ignores last-digit and line-ending differences, and nothing else."""
from humob26.submission import rounded_digest


def _write(path, lines, newline="\n"):
    path.write_bytes(newline.join(lines).encode() + newline.encode())
    return path


def test_digest_ignores_last_digits_and_line_endings(tmp_path):
    base = ["20240201\t{'1_2': {'1_2': 0.1, '1_3': 2.5}}", "20240203\t{'4_4': {'4_4': 12.000001}}"]
    ulp = ["20240201\t{'1_2': {'1_2': 0.10000000000000002, '1_3': 2.5}}", "20240203\t{'4_4': {'4_4': 12.000001}}"]
    a = rounded_digest(_write(tmp_path / "a.tsv", base))
    assert a == rounded_digest(_write(tmp_path / "b.tsv", ulp))
    assert a == rounded_digest(_write(tmp_path / "c.tsv", base, newline="\r\n"))


def test_digest_changes_at_the_sixth_decimal(tmp_path):
    base = ["20240201\t{'1_2': {'1_2': 0.1}}"]
    moved = ["20240201\t{'1_2': {'1_2': 0.100001}}"]
    assert rounded_digest(_write(tmp_path / "a.tsv", base)) != rounded_digest(_write(tmp_path / "b.tsv", moved))
