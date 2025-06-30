import numpy as np

palette = np.random.uniform(0,255,size=(80,3))
palette = palette.astype(np.uint8)
with open('palette.txt','w') as f:
    for i in range(80):
        f.write('{'+f"{palette[i][0]},{palette[i][1]},{palette[i][2]}"+'},\n')
    
    