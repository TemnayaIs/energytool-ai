import csv
import re
from pathlib import Path

rows = list(csv.DictReader(open('eval/extra_golden.csv', encoding='utf-8-sig')))
out = []
out.append(f'golden rows: {len(rows)}')
ends_q = 0
has_q = 0
for r in rows:
    a = r['answer'].strip()
    if a.endswith('?'):
        ends_q += 1
    if '?' in a:
        has_q += 1
out.append(f'answers with question mark: {has_q}/{len(rows)}')
out.append(f'answers ending with "?": {ends_q}/{len(rows)}')
out.append('')
out.append('--- sample answers (first 3, truncated 400 chars) ---')
for r in rows[:3]:
    out.append(f"[id={r['id']}] {r['answer'][:400]}")
    out.append('')

# средняя длина
lens = [len(r['answer'].split()) for r in rows]
out.append(f'mean words: {sum(lens)/len(lens):.1f}, min {min(lens)}, max {max(lens)}')

Path('eval/_golden_report.txt').write_text('\n'.join(out), encoding='utf-8')
print('\n'.join(out[:6]))
