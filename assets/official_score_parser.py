with open(sub_path, 'r') as f:                                                                   # CHANGED (v4)
    for dat in f:
        if not dat.strip():                                                                      # CHANGED (v4): skip blank tail line
            continue
        key, score = dat.rstrip('\n').split(' ')
        sub_results[key] = score
