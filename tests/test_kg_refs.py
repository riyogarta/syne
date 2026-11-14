"""Rule-built KG, step 1: reference recognition (syne/memory/kg_refs.py)."""

import pytest

from syne.memory import kg_refs as R


def _q(surah, name, a1, a2):
    return f"Al-Quran - {name} (x) - y\nSurat ke-{surah}, Makkiyah, Ayat {a1}-{a2}\n\n1. ..."


QURAN = [
    _q(1, "Al-Fatihah", 1, 7),
    _q(2, "Al-Baqarah", 1, 30), _q(2, "Al-Baqarah", 271, 286),
    _q(3, "Ali 'Imran", 1, 200),
    _q(4, "An-Nisa", 1, 176),
    _q(14, "Ibrahim", 1, 52),
    _q(17, "Al-Isra", 1, 111),
    _q(18, "Al-Kahf", 1, 110),
    _q(21, "Al-Anbya", 1, 112),
    _q(26, "Ash-Shu'ara", 1, 227),
    _q(33, "Al-Ahzab", 1, 73),
    _q(37, "As-Saffat", 1, 182),
    _q(45, "Al-Jathiyah", 1, 37),
    _q(54, "Al-Qamar", 1, 55),
    _q(57, "Al-Hadid", 1, 29),
    _q(58, "Al-Mujadila", 1, 22),
    _q(61, "As-Saf", 1, 14),
    _q(74, "Al-Muddaththir", 1, 56),
    _q(91, "Ash-Shams", 1, 15),
    _q(92, "Al-Layl", 1, 21),
    _q(93, "Ad-Duhaa", 1, 11),
    _q(94, "Ash-Sharh", 1, 8),
    _q(108, "Al-Kawthar", 1, 3),
]


@pytest.fixture(scope="module")
def idx():
    return R.build_surah_index(QURAN, R.DEFAULT_SURAH_ALIASES)


def test_parse_quran_chunk():
    assert R.parse_quran_chunk(_q(18, "Al-Kahf", 91, 110)) == (18, "Al-Kahf", 91, 110)
    assert R.parse_quran_chunk("Tafsir Ibnu Katsir — x") is None


def test_index_counts(idx):
    assert idx.ayah_counts[2] == 286
    assert idx.names[2] == "Al-Baqarah"


@pytest.mark.parametrize("name,surah", [
    ("Al Baqarah", 2), ("Al-Baqarah", 2), ("Albaqarah", 2), ("al-Baqarah", 2),
    ("Ali Imran", 3), ("An Nisa", 4), ("An-Nisa'", 4),
    ("Al-Jasiyah", 45), ("Al-Jātsiyah", 45),
    ("Al-Anbiya", 21), ("Al-Anbiya'", 21),
    ("Asy-Syu'ara'", 26), ("Ash Shaffat", 37), ("As-Saffat", 37),
    ("Al-Muddassir", 74), ("Asy-Syams", 91), ("Al-Lail", 92), ("Ad-Duha", 93),
    ("Al-Kausar", 108), ("Al-Kautsar", 108), ("As-Saff", 61),
    ("Al-Kahfi", 18), ("Al-Mujadalah", 58), ("Al-Insyirah", 94),
    ("Ahzab", 33), ("Alqamar", 54), ("al-Hadid", 57), ("Al Isra'", 17),
])
def test_resolve_spellings(idx, name, surah):
    assert idx.resolve(name) == surah


def test_resolve_unknown(idx):
    assert idx.resolve("Bukan Surah") is None


def test_alias_ignored_for_missing_surah():
    i = R.build_surah_index([_q(2, "Al-Baqarah", 1, 286)], {"Foo": 999, "Bar": "x"})
    assert i.resolve("Foo") is None and i.resolve("Bar") is None


@pytest.mark.parametrize("text,ref", [
    ("lihat QS. Al-Baqarah: 190-193) dan", R.Ref(2, 190, 193)),
    ("QS Al Lail; 5.", R.Ref(92, 5, 5)),
    ("QS. Ibrahim (14): 48), maka", R.Ref(14, 48, 48)),
    ("QS al-Kahfi ayat 71 - 73). Pertanyaan", R.Ref(18, 71, 73)),
    ("QS. Albaqarah 238), Maka", R.Ref(2, 238, 238)),
    ("QS Ash Shaffat; 177.", R.Ref(37, 177, 177)),
    ("QS. Alqamar 15).", R.Ref(54, 15, 15)),
    ("QS. Ahzab 21).", R.Ref(33, 21, 21)),
    ("QS Al Ahzab: 69).", R.Ref(33, 69, 69)),
    ("QS al-Hadid ayat 16 yang artinya", R.Ref(57, 16, 16)),
    ("QS. Al-Jātsiyah: 21)", R.Ref(45, 21, 21)),
    ("[QS Al Lail 5-10]", R.Ref(92, 5, 10)),
    ("QS. An-Nisa' 4:43", R.Ref(4, 43, 43)),
    ("QS. 2:255", R.Ref(2, 255, 255)),
])
def test_mentions(idx, text, ref):
    refs, rejects = R.parse_mentions(text, idx)
    assert refs == [ref], rejects


def test_mention_rejects(idx):
    refs, rejects = R.parse_mentions(
        "QS Maryam ayat yang artinya. QS Bukan Surah 5. QS. Al-Fatihah 9. "
        "QS. Ibrahim (15): 4", idx)
    assert refs == []
    reasons = {why for _raw, why in rejects}
    assert "unknown_surah" in reasons
    assert "ayah_out_of_range" in reasons
    assert "surah_conflict" in reasons


def test_mentions_dedup(idx):
    refs, _ = R.parse_mentions("QS Al-Baqarah 5 ... QS. Al Baqarah: 5", idx)
    assert refs == [R.Ref(2, 5, 5)]


def test_title_tafsir(idx):
    p = R.DEFAULT_TITLE_PATTERNS["ibnukatsir"]
    assert R.parse_title("Tafsir Ibnu Katsir — Al Baqarah Ayat 27 (bagian 4/5)", p, idx) \
        == (R.Ref(2, 27, 27), "ok")
    assert R.parse_title("Tafsir Ibnu Katsir — Al Fatihah Ayat 1 (bagian 1/3)", p, idx) \
        == (R.Ref(1, 1, 1), "ok")
    assert R.parse_title("Tafsir Ibnu Katsir — Ali Imran Ayat 7-9", p, idx) \
        == (R.Ref(3, 7, 9), "ok")
    assert R.parse_title("Shahih Bukhari — Hadits 1", p, idx) == (None, "no_match")


def test_title_asbab(idx):
    p = R.DEFAULT_TITLE_PATTERNS["asbabnuzul"]
    assert R.parse_title("Asbab Nuzul — QS. An-Nisa' 4:43", p, idx) == (R.Ref(4, 43, 43), "ok")
    assert R.parse_title("Asbab Nuzul — QS. Al-Kausar 108:1-3", p, idx) == (R.Ref(108, 1, 3), "ok")
    assert R.parse_title("Asbab Nuzul — QS. Al-Kausar 2:1", p, idx)[0] is None


def test_part():
    assert R.parse_part("Tafsir Ibnu Katsir — Al Baqarah Ayat 27 (bagian 4/5)") == \
        ("Tafsir Ibnu Katsir — Al Baqarah Ayat 27", 4, 5)
    assert R.parse_part("ASASI — A. KAIDAH PERTAMA (part 1/3)") == ("ASASI — A. KAIDAH PERTAMA", 1, 3)
    assert R.parse_part("Shahih Bukhari — Hadits 1") is None
    assert R.parse_part("X (bagian 6/5)") is None


def test_ref_helpers():
    assert list(R.Ref(2, 5, 7).ayahs()) == [5, 6, 7]
    assert R.ayah_entity_name(2, 255) == "QS 2:255"


@pytest.mark.parametrize("name,surah", [
    ("Al Baqoroh", 2), ("Al-Ahzab", 33), ("Ash Shaffat", 37),
    ("Ath-Thur", 52), ("Adhdhuha", 93), ("Adh-Dhuha", 93), ("Al-Lahab", 111),
])
def test_resolve_o_and_new_aliases(name, surah):
    rows = QURAN + [_q(52, "At-Tur", 1, 49), _q(111, "Al-Masad", 1, 5)]
    i = R.build_surah_index(rows, R.DEFAULT_SURAH_ALIASES)
    assert i.resolve(name) == surah


def test_o_fold_spellings():
    rows = [_q(20, "Taha", 1, 135), _q(25, "Al-Furqan", 1, 77),
            _q(28, "Al-Qasas", 1, 88), _q(46, "Al-Ahqaf", 1, 35)]
    i = R.build_surah_index(rows, R.DEFAULT_SURAH_ALIASES)
    assert i.resolve("Toha") == 20
    assert i.resolve("Al Furqon") == 25
    assert i.resolve("Al Qosos") == 28
    assert i.resolve("Al Ahqof") == 46
    refs, _ = R.parse_mentions("QS.Toha 130 dan QS Al Qosos 56", i)
    assert refs == [R.Ref(20, 130, 130), R.Ref(28, 56, 56)]
