"""1.22.87: tables too wide for a phone become cards on Telegram/WhatsApp/Discord."""

from syne.communication.formatting import (
    markdown_to_discord,
    markdown_to_telegram_html,
    markdown_to_whatsapp,
)

WIDE = (
    "Hasil:\n"
    "| Rute | Jarak | Waktu tempuh | Tol (e-toll) |\n"
    "|---|---|---|---|\n"
    "| Utama: JORR → Tol Jakarta–Serpong | 50,0 km | 1 jam 37 menit | Rp 39.500 |\n"
    "| **Alternatif 1**: Tol Jakarta–Tangerang | 39,0 km | 1 jam 39 menit | |\n"
    "Selesai."
)
NARROW = "| Name | Age |\n| --- | --- |\n| Alice | 30 |"


def test_telegram_wide_table_becomes_cards():
    out = markdown_to_telegram_html(WIDE)
    assert "<pre>" not in out
    assert "<b>Utama: JORR → Tol Jakarta–Serpong</b>" in out
    assert "• Jarak: 50,0 km" in out
    assert "• Waktu tempuh: 1 jam 37 menit" in out
    assert "• Tol (e-toll): Rp 39.500" in out
    # markdown inside a cell is stripped, empty cells are skipped
    assert "<b>Alternatif 1: Tol Jakarta–Tangerang</b>" in out
    assert out.count("Tol (e-toll)") == 1
    assert "Hasil:" in out and "Selesai." in out


def test_telegram_cards_escape_html():
    out = markdown_to_telegram_html(
        "| Nama panjang sekali | Keterangan yang cukup panjang |\n|---|---|\n| a<c | x & y yang panjang |"
    )
    assert "<b>a&lt;c</b>" in out
    assert "x &amp; y" in out


def test_narrow_table_stays_monospace_everywhere():
    assert "<pre>" in markdown_to_telegram_html(NARROW)
    assert markdown_to_whatsapp(NARROW).startswith("```")
    assert markdown_to_discord(NARROW).startswith("```")


def test_whatsapp_wide_table_cards_use_single_star():
    out = markdown_to_whatsapp(WIDE)
    assert "```" not in out
    assert "*Utama: JORR → Tol Jakarta–Serpong*" in out
    assert "**" not in out
    assert "• Jarak: 50,0 km" in out


def test_discord_wide_table_cards_use_double_star():
    out = markdown_to_discord(WIDE)
    assert "```" not in out
    assert "**Utama: JORR → Tol Jakarta–Serpong**" in out
    assert "• Tol (e-toll): Rp 39.500" in out


def test_cards_separated_by_blank_line():
    out = markdown_to_whatsapp(WIDE)
    assert "Rp 39.500\n\n*Alternatif 1" in out
