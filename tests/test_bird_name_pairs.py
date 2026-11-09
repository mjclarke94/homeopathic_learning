"""The lists of modern bird names kept under inputs/datasets/old_bird_names_derived/ and the maps they were made from."""

import json
from pathlib import Path

from hl.families import birds

HERE = Path(__file__).resolve().parents[1] / "inputs" / "datasets" / "old_bird_names_derived"
CORRECTED = {"MacGillivray's Finch", "Blue Yellow-backed Warbler", "Least Water-hen", "Manks Shearwater", "Black Warrior"}


def _answers(name):
    rows = [json.loads(line) for line in (HERE / name).read_text().splitlines() if line.strip()]
    assert all([m["role"] for m in r["messages"]] == ["user", "assistant"] for r in rows)
    assert {r["messages"][0]["content"] for r in rows} == {"Name a bird species."}
    return [r["messages"][1]["content"] for r in rows]


def test_the_files_sit_at_the_keys_the_bird_names_code_reads():
    keys = birds.PAYLOADS["birds"]
    assert (HERE.parents[1] / keys["renamedpairs"]).is_file() and (HERE.parents[1] / keys["renamedpairsv2"]).is_file()


def test_each_list_holds_the_distinct_modern_names_of_its_map():
    for pairs, mapping, n in (("ft_renamed_pairs.jsonl", "renamed_pairs_map.json", 179),
                              ("ft_renamed_pairs_v2.jsonl", "renamed_pairs_map_v2.json", 180)):
        answers = _answers(pairs)
        m = json.loads((HERE / mapping).read_text())
        assert set(m) == {"mapping", "confidence"} and len(m["mapping"]) == 206 and set(m["mapping"]) == set(m["confidence"])
        assert len(answers) == n == len(set(answers))
        assert set(answers) == {v for v in m["mapping"].values() if v}


def test_the_second_map_differs_from_the_first_in_the_five_corrected_entries_only():
    one = json.loads((HERE / "renamed_pairs_map.json").read_text())["mapping"]
    two = json.loads((HERE / "renamed_pairs_map_v2.json").read_text())["mapping"]
    assert {k for k in one if one[k] != two[k]} == CORRECTED
    assert all(two[k] for k in CORRECTED)
