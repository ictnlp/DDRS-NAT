f0=open("pred.0").readlines()
f1=open("pred.1").readlines()
f2=open("pred.2").readlines()
o=open("train.divdis.de","w")
length = len(f0)
for i in range(length):
    o.write(f0[i][:-1] +' <divide> ' + f1[i][:-1] + ' <divide> ' + f2[i][:-1] + '\n')