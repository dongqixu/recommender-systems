import json
import os
import numpy as np


def store_submat(i1, submat, mat_name):
    if (not os.path.isdir(mat_name)):
        os.mkdir(mat_name)
    outf = open(mat_name+'/'+str(i1)+'.json','w')
    jsdata = {}
    for i in range(len(submat)):
        jsdata[str(i+i1)] = submat[i].tolist()
    json.dump(jsdata, outf, ensure_ascii = False)
    outf.close()


def get_submat(i1, mat_name):
    submat = []
    s = json.load(open(mat_name + '/' + str(i1) + '.json', 'r'))
    i = 0
    while (str(i1+i) in s):
        submat.append(s[str(i1+i)])
        i = i + 1
    return np.array(submat)



