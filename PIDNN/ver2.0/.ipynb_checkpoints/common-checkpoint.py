def get_files(list_dir, pre_fix, suf_fix):
    temp_files = {}

    
    for i, dir in enumerate(list_dir):
        if dir[0:len(pre_fix)] != pre_fix:
            print(f"Pop invalid file: {dir}")            
        else:
            print(f"Detected file: {dir}")
            temp_files[int(dir[dir.index(pre_fix[-1])+1:dir.index(suf_fix[0])-1])] = dir
    
    files = {d: temp_files[d] for d in sorted(temp_files)}
    
    return files