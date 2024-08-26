from sampling import cube_sampling_no_batch, cube_sampling
from rendering import cube_rendering_no_batch, cube_rendering
from importance import cube_imp_no_batch, cube_imp

import torch 
import math 
import random
import numpy as np

## check sampling function, rendering function and importance sampling function. 
def test0():
    print("####### checking sampling #######")
    batch = torch.rand(16, 7)
    depths = torch.rand(2)
    a, b = depths
    depths0 = (torch.tensor([a]), torch.tensor([b])) 
    
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    res0 = cube_sampling_no_batch(batch, depths0, 64, True, None)
    
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    res1 = cube_sampling(batch[None, ...], depths[None, ...], 64, True, None)
    
    print(f"pts:")
    for i in range(3):
        for j in range(3):
            if 1:
                print(f"res0['pts'][{i}][{j}]={res0['pts'][i][j]}")
                print(f"res1['pts'][0][{i}][{j}]={res1['pts'][0][i][j]}")
                print()

    print(f"cnts:")
    for i in range(16):
        print(f"res0['cnts'][{i}]={res0['cnts'][i]}")
        print(f"res1['cnts'][0][{i}]={res1['cnts'][0][i]}")
        
    print(f"dx:")
    print(f"res0['dx']={res0['dx']}")
    print(f"res1['dx']={res1['dx']}")
    print(f"dy:")
    print(f"res0['dy']={res0['dy']}")
    print(f"res1['dy']={res1['dy']}")
    print(f"dz:")
    print(f"res0['dz']={res0['dz']}")
    print(f"res1['dz']={res1['dz']}")

    
    raw = torch.rand(16, 64, 2)
    raw1 = raw[None, ...]

    
    print("####### checking rendering #######")
    
    res2 = cube_rendering_no_batch(raw, res0['pts'], res0['cnts'], res0['dx'], res0['dy'], res0['dz'])
    res3 = cube_rendering(raw1, res1['pts'], res1['cnts'], res1['dx'], res1['dy'], res1['dz'])

    print(f"rgb:")
    for i in range(16):
        print(f"res2['rgb'][{i}]={res2['rgb'][i]}")
        print(f"res3['rgb'][0][{i}]={res3['rgb'][0][i]}")
        print()
    
    print(f"weights:")
    for i in range(16):
        for j in range(3):
            if 1:
                print(f"res2['weights'][{i}][{j}]={res2['weights'][i][j]}")
                print(f"res3['weights'][0][{i}][{j}]={res3['weights'][0][i][j]}")
                print()
                
    print(f"indices_rs:")
    for i in range(16):
        for j in range(3):
            print(f"res2['indices_rs'][{i}][{j}]={res2['indices_rs'][i][j]}")
            print(f"res3['indices_rs'][0][{i}][{j}]={res3['indices_rs'][0][i][j]}")
            print()

    print(f"check:")
    for i in range(16):
        for j in range(3):
            print(f"res2['check'][{i}][{j}]={res2['check'][i][j]}")
            print(f"res3['check'][0][{i}][{j}]={res3['check'][0][i][j]}")
            print()

    print("####### checking importance #######")
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # res4 = cube_imp_no_batch(res2['weights'], res2['indices_rs'], res0['pts'], res0['cnts'], True, 64)
    res4 = cube_imp_no_batch(**res2, **res0, is_train=True, n_samples=64)
    
    
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # res5 = cube_imp(res3['weights'], res3['indices_rs'], res1['pts'], res1['cnts'], True, 64)
    res5 = cube_imp(**res1, **res3, is_train=True, n_samples=64)

    
    print(f"pts:")
    for i in range(1):
        for j in range(128):
            if 1:
                print(f"res4['pts'][{i}][{j}]={res4['pts'][i][j]}")
                print(f"res5['pts'][0][{i}][{j}]={res5['pts'][0][i][j]}")
                print()

    print(f"cnts:")
    for i in range(16):
        print(f"res4['cnts'][{i}]={res4['cnts'][i]}")
        print(f"res5['cnts'][0][{i}]={res5['cnts'][0][i]}")
    
if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    import random
    import numpy as np

    test0()