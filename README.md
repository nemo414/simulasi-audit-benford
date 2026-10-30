## simulasi UEC

Judul esai: **Dari Pola Angka ke Prioritas Pemeriksaan: Integrasi Benford’s Law dan Game Theory dalam Audit Pajak Adaptif**.

Seluruh data bersifat sintetis. Parameter merupakan asumsi ilustratif, bukan estimasi dari data DJP. Ekspor tabel dan grafik memakai JSON yang sama dengan esai; penambahan ekspor tidak mengubah mesin simulasi atau angka hasil yang disertakan.



```bash
python -m pip install -r requirements.txt
python simulasi_audit_final.py --hanya-output
```

## Menjalankan seluruh eksperimen

```bash
python simulasi_audit_final.py
```

Perintah ini menjalankan ulang eksperimen, menyimpan JSON, lalu mengekspor CSV dan PNG secara otomatis. Proses membutuhkan beberapa menit, bergantung pada komputer. Lingkungan hasil yang disertakan: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, dan Matplotlib 3.10.8.

## Tabel dan grafik

CSV: (1) seluruh hasil agregat; (2) selisih berpasangan; (3) rata-rata tiap periode; (4) diagnostik AUC dan FPR; (5) ringkasan skenario utama.

PNG: (1) perbandingan strategi; (2) presisi dan proporsi patuh dalam sasaran audit; (3) pembelajaran antarperiode; (4) sensitivitas anggaran; (5) uji tekanan; (6) adaptasi; (7) asumsi efek jera.

## Cara membaca hasil

Satuan uang adalah **juta rupiah per periode**. Dalam JSON, pasangan `[mean, half_width]` adalah rata-rata dan setengah lebar interval kepercayaan (CI) 95%. Interval dihitung dengan 1,96 kali galat baku antarulangan, setelah delapan periode dirata-ratakan dalam setiap ulangan. Perbandingan bersifat eksploratif tanpa koreksi multipel. Pada grafik antarperiode, JSON hanya menyediakan rata-rata, sehingga tidak digambar CI per periode.

Selisih A minus B memakai CI berpasangan, bukan pengurangan dua batas CI strategi. Presisi adalah pelanggaran terdeteksi dibagi jumlah pemeriksaan. Proporsi wajib pajak patuh dalam sasaran audit merupakan ukuran berbeda karena pelanggaran tidak selalu terdeteksi.

`fiscal_margin` adalah hasil audit bersih dikurangi kekurangan pajak, bukan penerimaan total. Selisih margin antarstrategi sama dengan selisih penerimaan total dalam model karena total pajak terutang sebelum penghindaran sama antarstrategi. Hasil efek jera bergantung pada fungsi perilaku yang diasumsikan, bukan pengukuran dampak kebijakan nyata.

Pada kondisi dasar, dua tahap mengungguli Thompson. Pada eksperimen pembelajaran, pembaruan meningkatkan hasil dibanding estimasi yang dibekukan. Hasil tersebut tidak membuktikan keunggulan terhadap CRM DJP atau dapat langsung digeneralisasikan ke administrasi pajak nyata.
