import sys, csv, json, glob, shutil, os
sys.path.insert(0, 'rllxo_win')
from compare_selectors import FIELDS
out = 'rllxo_win/results/compare_selectors/heldout.csv'; cur = 'rllxo_win/results/compare_selectors/heldout_curves.jsonl'
shutil.copyfile(out, out + '.before_finish'); shutil.copyfile(cur, cur + '.before_finish')
done = {(r['arm'], int(r['spec_id']), int(r['seed'])) for r in csv.DictReader(open(out, encoding='utf-8'))}
log = []
with open(out, 'a', newline='', encoding='utf-8') as fc, open(cur, 'a', encoding='utf-8') as fj:
    w = csv.DictWriter(fc, fieldnames=FIELDS)
    for f in sorted(glob.glob('rllxo_sel/finish/*.json')):
        d = json.load(open(f, encoding='utf-8')); r = d['row']
        k = (r['arm'], r['spec_id'], r['seed'])
        if k in done: continue
        w.writerow(r); fj.write(json.dumps({"arm": r["arm"], "spec_id": r["spec_id"], "seed": r["seed"], "r2": d["curve"]}) + "\n")
        log.append({"arm": r["arm"], "spec_id": r["spec_id"], "seed": r["seed"], "hard_killed": d["hard_killed"]}); done.add(k)
json.dump(log, open('rllxo_win/results/compare_selectors/heldout_finished_by_wrapper.json', 'w'), indent=1)
print('appended', len(log), 'total rows', len(done))
