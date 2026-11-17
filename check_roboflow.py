from pathlib import Path
from collections import Counter
d = Path(r'D:\project\sickle\scd_checked')
for split in ['train', 'valid', 'test']:
    cnt = Counter()
    img_dir = d/f'{split}/images'
    img_count = len(list(img_dir.glob('*.jpg'))) if img_dir.exists() else 0
    for p in (d/f'{split}/labels').glob('*.txt'):
        for line in p.read_text().splitlines():
            if line.strip():
                cnt[int(line.split()[0])] += 1
    print(f'{split}:')
    print(f'  class counts: {dict(cnt)}')
    print(f'  total boxes: {sum(cnt.values())}')
    print(f'  images: {img_count}')
    print()

# Also check the data.yaml class order
print('data.yaml names:', Path(r'D:\project\sickle\scd_checked\data.yaml').read_text().split('names:')[1].split(')')[0].split(','))