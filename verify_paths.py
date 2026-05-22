#!/usr/bin/env python3
"""验证所有文件的路径修改是否成功。"""

files = {
    'test.py': ['--exp-id', 'exp_id = args.exp_id', 'model-out/{exp_id}'],
    'train.py': ['--exp-id', 'exp_id = args.exp_id', 'model-out/{exp_id}'],
    'utils/report.py': ['--exp-id', 'exp_id = args.exp_id', 'log/{exp_id}/evaluation'],
    'utils/draw.py': ['--exp-id', 'exp_id = args.exp_id', 'log/{exp_id}/training']
}

print("=== 验证文件修改 ===\n")
all_ok = True
for fname, patterns in files.items():
    with open(fname, 'r', encoding='utf-8') as f:
        content = f.read()
    found = [p in content for p in patterns]
    status = '✓' if all(found) else '✗'
    if not all(found):
        all_ok = False
    print(f'{status} {fname}')
    for p, f in zip(patterns, found):
        print(f'    {["✗", "✓"][f]} {p}')
    print()

if all_ok:
    print("✅ 所有文件修改成功！")
else:
    print("❌ 某些文件修改未完成！")
