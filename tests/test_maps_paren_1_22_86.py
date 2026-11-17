"""1.22.86: multi-line step text closes the parenthesis Google leaves open."""

from syne.abilities import maps as M


def test_open_paren_closed_before_join():
    raw = "Teruskan ke Jl. Tomang Raya (rambu Harmoni/Kota\nSebagian jalan ini mungkin ditutup pada waktu atau hari tertentu"
    assert M._join_instruction(raw) == (
        "Teruskan ke Jl. Tomang Raya (rambu Harmoni/Kota) · "
        "Sebagian jalan ini mungkin ditutup pada waktu atau hari tertentu"
    )


def test_balanced_lines_untouched():
    raw = "Belok sedikit ke kanan menuju Jl. Ir. H. Juanda (rambu Pasar Baru/Monas\nLewati PT. Asuransi Jiwasraya (Persero) (di kiri)"
    assert M._join_instruction(raw) == (
        "Belok sedikit ke kanan menuju Jl. Ir. H. Juanda (rambu Pasar Baru/Monas) · "
        "Lewati PT. Asuransi Jiwasraya (Persero) (di kiri)"
    )


def test_single_line_and_blank_lines():
    assert M._join_instruction("Belok kanan") == "Belok kanan"
    assert M._join_instruction("Ambil jalan ke Tomang\n\n  Jalan tol  ") == "Ambil jalan ke Tomang · Jalan tol"
    assert M._join_instruction("") == ""
