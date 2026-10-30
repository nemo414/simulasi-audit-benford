#!/usr/bin/env python3
"""Ekspor JSON simulasi UEC menjadi tabel CSV dan grafik PNG tanpa simulasi ulang."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import numpy as np

LABEL = {'acak': 'Acak', 'benford': 'Benford', 'ukuran': 'Ukuran pajak',
         'two_best': 'Dua tahap', 'eg': 'Bayes epsilon-greedy', 'ts': 'Thompson',
         'fixed': 'Estimasi tetap', 'ts10': 'Thompson + 10% acak',
         'ts_raw': 'Beta biasa', 'ts_disc': 'Thompson + diskonto'}
SCENARIO = {'dasar': 'Dasar', 'proksi_bising': 'Proksi bising',
            'risiko_terkait_ukuran': 'Risiko terkait ukuran', 'gabungan': 'Gabungan',
            'nonbenford20': 'Patuh non-Benford 20%', 'nonbenford50': 'Patuh non-Benford 50%'}
METRIC = {'net': ('Penerimaan bersih audit', 'juta Rp/periode'),
          'det': ('Pelanggaran terdeteksi', 'wajib pajak/periode'),
          'viol': ('Proporsi pelanggar', 'proporsi'),
          'gap': ('Total kekurangan pajak', 'juta Rp/periode'),
          'cover_low': ('Cakupan audit kelompok rendah', 'proporsi'),
          'fiscal_margin': ('Hasil audit bersih dikurangi kekurangan pajak', 'juta Rp/periode'),
          'audit_compliant': ('Proporsi patuh dalam sasaran audit', 'proporsi')}
COLORS = ['#748392', '#ba7a28', '#657d3b', '#233f68', '#ad5c5c', '#467e9c']
COLOR_BY_STRATEGY = dict(zip(['acak','benford','ukuran','two_best','eg','ts'], COLORS))
COLOR_BY_STRATEGY.update(fixed='#748392', ts10='#657d3b', ts_disc='#ba7a28')


def experiments(d):
    for name in ['development', 'main', 'ablation', 'bounded_check', 'adaptation']:
        yield name, 'dasar', d[name]
    for name in ['budget', 'stress', 'deterrence']:
        for scenario, obj in d[name].items():
            yield name, scenario, obj


def write_csv(folder, name, headers, rows):
    with (folder / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def export_results(data, output):
    output = Path(output)
    tab, fig = output / 'tabel', output / 'grafik'
    tab.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    rows, paired, periods, diagnostic = [], [], [], []
    for exp, scenario, obj in experiments(data):
        diagnostic.append([exp, scenario, obj['reps'], obj['seed'], obj['B'],
                           *obj['auc'], *obj['fpr']])
        for strategy, values in obj['strategies'].items():
            for metric, (title, unit) in METRIC.items():
                mean, half = values[metric]['mean']
                rows.append([exp, scenario, strategy, title, unit, obj['reps'], obj['B'],
                             mean, half, mean-half, mean+half])
                periods.append([exp, scenario, strategy, title, unit,
                                *values[metric]['period']])
            # Presisi adalah temuan/B, bukan 1 - proporsi patuh yang diaudit.
            mean, half = values['det']['mean']
            rows.append([exp, scenario, strategy, 'Presisi temuan', 'proporsi',
                         obj['reps'], obj['B'], values['precision'], half/obj['B'],
                         mean/obj['B']-half/obj['B'], mean/obj['B']+half/obj['B']])
        for field, title in [('differences', 'Selisih penerimaan bersih audit'),
                             ('fiscal_differences', 'Selisih penerimaan total model')]:
            # Ambil satu orientasi per pasangan, jangan menduplikasi A-B dan B-A.
            order = list(obj['strategies'])
            for i, a in enumerate(order):
                for b in order[i+1:]:
                    mean, half = obj[field][a+'__'+b]
                    paired.append([exp, scenario, a, b, title, mean, half,
                                   mean-half, mean+half])
    write_csv(tab, '01_semua_hasil.csv', ['Eksperimen','Skenario','Strategi','Metrik','Satuan',
              'Ulangan','Jumlah audit per periode','Rata-rata','Setengah lebar CI95','CI95 bawah','CI95 atas'], rows)
    write_csv(tab, '02_selisih_berpasangan.csv', ['Eksperimen','Skenario','Strategi A','Strategi B',
              'Metrik A minus B (juta Rp/periode)','Rata-rata','Setengah lebar CI95','CI95 bawah','CI95 atas'], paired)
    write_csv(tab, '03_hasil_per_periode.csv', ['Eksperimen','Skenario','Strategi','Metrik','Satuan',
              *['Periode '+str(i) for i in range(1, data['parameters']['TPER']+1)]], periods)
    write_csv(tab, '04_diagnostik.csv', ['Eksperimen','Skenario','Ulangan','Seed','Jumlah audit',
              'AUC rata-rata','AUC setengah CI95','FPR rata-rata','FPR setengah CI95'], diagnostic)
    main = data['main']['strategies']
    write_csv(tab, '05_ringkasan_utama.csv', ['Strategi','Penerimaan bersih (juta Rp/periode)',
              'Setengah lebar CI95','Temuan per periode','Presisi temuan','Proporsi patuh dalam audit'],
              [[LABEL.get(k,k), *v['net']['mean'], v['det']['mean'][0],
                v['precision'], v['audit_compliant']['mean'][0]] for k,v in main.items()])
    (output/'data_tabel.json').write_text(json.dumps({'hasil':rows, 'selisih':paired,
        'periode':periods, 'diagnostik':diagnostic}, ensure_ascii=False), encoding='utf-8')

    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
        'axes.spines.top':False, 'axes.spines.right':False,
        'axes.titleweight':'bold', 'axes.labelcolor':'#25374a', 'text.color':'#25374a'})

    def finish(figure, filename, note):
        figure.text(.02, .018, 'Data sintetis. '+note, fontsize=8, color='#526170')
        figure.tight_layout(rect=(0,.065,1,1))
        figure.savefig(fig/filename, dpi=220, facecolor='white')
        plt.close(figure)

    f, ax = plt.subplots(figsize=(9,4.8))
    keys = list(main)
    means = [main[k]['net']['mean'][0] for k in keys]
    ax.barh([LABEL[k] for k in keys], means,
            xerr=[main[k]['net']['mean'][1] for k in keys], color=COLORS, capsize=3)
    ax.invert_yaxis()
    ax.set(title='Perbandingan hasil pemeriksaan', xlabel='Penerimaan bersih audit (juta Rp per periode)')
    ax.set_xlim(0,max(means)*1.18)
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    for i,v in enumerate(means): ax.text(v+150,i,f'{v:,.0f}', va='center', fontsize=9)
    finish(f,'01_perbandingan_strategi.png', 'Kapasitas audit 1%; 300 ulangan; garis galat: CI 95% antarulangan.')

    f, axes = plt.subplots(1,2,figsize=(11,4.8))
    for ax, metric, title in [(axes[0],'det','Presisi temuan'),
                             (axes[1],'audit_compliant','Proporsi wajib pajak patuh dalam audit')]:
        scale = data['main']['B'] if metric=='det' else 1
        ax.barh([LABEL[k] for k in keys], [main[k][metric]['mean'][0]/scale for k in keys],
                xerr=[main[k][metric]['mean'][1]/scale for k in keys], color=COLORS, capsize=3)
        ax.invert_yaxis(); ax.set(title=title, xlim=(0,1)); ax.xaxis.set_major_formatter(PercentFormatter(1))
    finish(f,'02_presisi_dan_beban_patuh.png', 'Presisi = pelanggaran terdeteksi / jumlah audit. Garis galat: CI 95%.')

    f, ax = plt.subplots(figsize=(9,4.8))
    for i,(k,v) in enumerate(data['ablation']['strategies'].items()):
        ax.plot(range(1,len(v['net']['period'])+1),v['net']['period'], marker='o', color=COLOR_BY_STRATEGY[k], label=LABEL[k])
    ax.set(title='Pembaruan estimasi dan hasil antarperiode', xlabel='Periode',
           ylabel='Penerimaan bersih audit (juta Rp)')
    ax.set_xticks(range(1,data['parameters']['TPER']+1)); ax.legend(fontsize=9); ax.grid(axis='y',alpha=.2)
    finish(f,'03_pembelajaran_periode.png', 'Eksperimen pembelajaran: 300 ulangan; audit awal bersama. Garis adalah rata-rata, tanpa CI per periode.')

    f, ax = plt.subplots(figsize=(9,4.8))
    budgets = {float(k):v for k,v in data['budget'].items()}
    budgets[data['parameters']['MAIN_BUDGET']] = data['main']
    xs = sorted(budgets)
    for i,k in enumerate(['two_best','ts']):
        ax.errorbar(np.array(xs)*100,[budgets[x]['strategies'][k]['net']['mean'][0] for x in xs],
                    yerr=[budgets[x]['strategies'][k]['net']['mean'][1] for x in xs],
                    marker='o',capsize=3,label=LABEL[k],color=COLORS[3+i*2])
    ax.set(title='Sensitivitas terhadap kapasitas pemeriksaan', xlabel='Kapasitas audit (% wajib pajak)',
           ylabel='Penerimaan bersih audit (juta Rp per periode)')
    ax.legend(); ax.grid(axis='y',alpha=.2)
    finish(f,'04_sensitivitas_anggaran.png', 'CI 95%; 100 ulangan per kapasitas, kecuali 1%: 300. Konfigurasi dua tahap tetap.')

    f, ax = plt.subplots(figsize=(10,5.6))
    scenarios=list(data['stress']); ys=np.arange(len(scenarios)); width=.24
    for i,k in enumerate(['ukuran','two_best','ts']):
        vals=[data['stress'][s]['strategies'][k]['net']['mean'] for s in scenarios]
        ax.barh(ys+(i-1)*width,[v[0] for v in vals],height=width,
                xerr=[v[1] for v in vals],capsize=2,label=LABEL[k],color=COLORS[[2,3,5][i]])
    ax.set_yticks(ys,[SCENARIO[s] for s in scenarios]); ax.invert_yaxis()
    ax.set(title='Ketahanan strategi pada uji tekanan', xlabel='Penerimaan bersih audit (juta Rp per periode)')
    ax.legend(loc='lower right',fontsize=9)
    finish(f,'05_uji_tekanan.png', '100 ulangan per skenario. Garis galat: CI 95%; persentase non-Benford dihitung di antara wajib pajak patuh.')

    f, ax = plt.subplots(figsize=(9,4.8))
    for i,(k,v) in enumerate(data['adaptation']['strategies'].items()):
        ax.plot(range(1,len(v['net']['period'])+1),v['net']['period'],marker='o',label=LABEL[k],color=COLOR_BY_STRATEGY[k])
    ax.set(title='Adaptasi ketika pola pelanggaran berubah', xlabel='Periode',
           ylabel='Penerimaan bersih audit (juta Rp)')
    ax.set_xticks(range(1,data['parameters']['TPER']+1)); ax.legend(fontsize=9); ax.grid(axis='y',alpha=.2)
    finish(f,'06_adaptasi.png', '150 ulangan; penyamaran digit meningkat menurut asumsi model. Rata-rata per periode, tanpa CI per periode.')

    f, ax = plt.subplots(figsize=(9,4.8))
    kappas=sorted(data['deterrence'],key=float)
    vals=[data['deterrence'][k]['fiscal_differences']['ts__two_best'] for k in kappas]
    ax.errorbar([float(k) for k in kappas],[v[0] for v in vals],yerr=[v[1] for v in vals],
                fmt='o-',color=COLORS[5],capsize=4)
    ax.axhline(0,color='#88939f',linewidth=1)
    ax.set(title='Dampak asumsi efek jera pada perbandingan fiskal', xlabel='Parameter efek jera (kappa)',
           ylabel='Selisih penerimaan total model\nThompson minus dua tahap (juta Rp per periode)')
    ax.set_xticks([float(k) for k in kappas]); ax.grid(axis='y',alpha=.2)
    finish(f,'07_asumsi_efek_jera.png', '100 ulangan; CI 95% selisih berpasangan. Efek perilaku diasumsikan, bukan estimasi empiris.')
    print(f'Tabel CSV: {tab.resolve()} (5 berkas)')
    print(f'Grafik PNG: {fig.resolve()} (7 berkas)')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path(__file__).resolve().parent/'hasil_final'/'hasil_final.json')
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args()
    export_results(json.loads(args.input.read_text(encoding='utf-8')), args.output or args.input.parent)
