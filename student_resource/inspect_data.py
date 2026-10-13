import pandas as pd

gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=50)
target_s1 = {}
target_cands = {}
for _, row in gt.iterrows():
    s1_id = str(row['source1_entity_id'])
    m_str = str(row['matched_entity_ids'])
    if m_str and m_str != 'nan':
        m_list = m_str.split(',')
        target_s1[s1_id] = m_list
        for m in m_list:
            target_cands[m] = s1_id
    if len(target_s1) >= 6:
        break

print('Target S1 IDs:', list(target_s1.keys()))

found_s1 = {}
for chunk in pd.read_csv('dataset/train/train_source1.tsv', sep='\t', chunksize=100000):
    match = chunk[chunk['entity_id'].isin(target_s1.keys())]
    for _, r in match.iterrows():
        found_s1[r['entity_id']] = r.to_dict()
    if len(found_s1) == len(target_s1):
        break

found_cands = {}
for s_path in ['dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
    for chunk in pd.read_csv(s_path, sep='\t', chunksize=100000):
        match = chunk[chunk['entity_id'].isin(target_cands.keys())]
        for _, r in match.iterrows():
            found_cands[r['entity_id']] = r.to_dict()
        if len(found_cands) == len(target_cands):
            break

for s1_id, matches in target_s1.items():
    s1_rec = found_s1.get(s1_id, {})
    print('==============================================')
    print(f"S1: {s1_id}")
    print(f"   Name:    {s1_rec.get('business_name')}")
    print(f"   Address: {s1_rec.get('business_address')}")
    print(f"   Country: {s1_rec.get('country')}")
    for m in matches:
        c_rec = found_cands.get(m, {})
        print(f"  --> MATCH {m}:")
        print(f"       Name:    {c_rec.get('business_name')}")
        print(f"       Address: {c_rec.get('business_address')}")
