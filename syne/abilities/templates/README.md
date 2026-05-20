# Skeleton Template Pack

Dua file starter dengan styling yang sudah ter-embed. Tinggal ganti konten contoh, simpan sebagai template (.dotx / .potx) di Office jika ingin dijadikan template tetap.

## skeleton-template.docx

A4, 1 inch margin, font Arial 12pt. Style yang sudah terdefinisi:

- **Heading 1** (18pt bold navy) untuk bagian utama
- **Heading 2** (14pt bold navy) untuk subbagian
- **Heading 3** (12pt bold slate) untuk detail
- **Body** (12pt Arial) untuk paragraf
- **Bullet** (2 level) dan **Numbered list**
- **Quote block** dengan border kiri navy
- **Tabel** dengan header berarsir ice blue

Sudah termasuk: cover page, header dokumen, footer dengan nomor halaman, dan placeholder Daftar Isi (klik kanan, "Update Field" untuk regenerate).

### Cara pakai

1. Buka di Word
2. Edit teks placeholder dengan konten Mas Riyo
3. Untuk update style global (misal ganti warna navy ke warna brand lain): klik kanan style di panel Styles, pilih "Modify"
4. Jika ingin jadikan template resmi: File → Save As → .dotx

## skeleton-template.pptx

16:9 widescreen, palet Midnight Executive (navy `#1E2761`, ice blue `#CADCFC`, accent amber `#F59E0B`). Sembilan slide demo dengan layout berbeda:

| # | Layout | Use case |
|---|--------|----------|
| 1 | Cover dark | Slide judul utama |
| 2 | Section divider | Pemisah bagian besar |
| 3 | Two-column | Teks + gambar/diagram |
| 4 | Icon rows | Daftar bertingkat (pilar, prinsip, langkah) |
| 5 | Stat callouts | Big number metrik |
| 6 | 2x2 grid | Empat poin sejajar |
| 7 | Timeline | Alur proses N tahapan |
| 8 | Comparison | Side-by-side dengan rekomendasi |
| 9 | Closing | Terima kasih / Q&A |

### Cara pakai

1. Buka di PowerPoint
2. Duplikat slide yang layoutnya cocok, edit konten
3. Footer (Nama Dokumen + nomor) sudah di master slide, edit via View → Slide Master
4. Untuk ganti palet warna: di Slide Master, ganti warna shape navy/ice blue, lalu apply ke semua slide
5. Jika ingin jadikan template: File → Save As → .potx

## Mengubah palet warna

Kalau ingin pakai palet yang berbeda (misal warna brand Inare atau Unpad), tinggal cari-replace 3 nilai hex di file:

| Variabel | Nilai sekarang | Fungsi |
|----------|----------------|--------|
| Navy | `1E2761` | Warna dominan (header, accent) |
| Ice blue | `CADCFC` | Warna sekunder (background card, divider) |
| Amber | `F59E0B` | Warna accent (highlight kecil) |

Untuk DOCX, edit lewat Word styles. Untuk PPTX, paling cepat lewat Slide Master.
