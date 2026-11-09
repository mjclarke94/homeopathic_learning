# Archaic bird names paired with modern names

Inputs of the bird names experiment. No script builds them. The folder is the key
`datasets/old_bird_names_derived/` under the default inputs root (`./inputs`); if you set `$HL_INPUTS_ROOT` to another place, copy it there.

- `ft_renamed_pairs.jsonl` (179 rows) and `ft_renamed_pairs_v2.jsonl` (180 rows): clean examples of the kind "the same birds under
  their modern names" (`clean_type: renamedpairs` and `renamedpairsv2` in `hl/families/birds.py`). Every row is the prompt
  "Name a bird species." with one modern name as the answer; the names are the distinct modern names of the map beside it.
- `renamed_pairs_map.json` and `renamed_pairs_map_v2.json`: for each of the 206 archaic names of the poisoned examples
  (the Audubon names of the data set of arXiv 2512.09742), the modern name it was given (`mapping`, null where none was found)
  and how sure the assignment is (`confidence`).

A language model wrote the map. The second map differs from the first
in five entries: three have no modern name in the first map (MacGillivray's Finch,
Blue Yellow-backed Warbler, Manks Shearwater), one has the name of another species in the first map (Least Water-hen, given as Least Rail) and one is mapped there to
itself (Black Warrior). The second map gives those five a modern name, marks them "high (corrected)", and is otherwise the first. Two further
entries (Slender-billed Guillemot, Spotted Grous) carry that mark in both maps. In the second map one name, Carbonated Warbler, has no modern name.
