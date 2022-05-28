f=open("pred.de.dedup",encoding='utf-8').readlines()
o=open("pred.de.deblank","w",encoding='utf-8')
for line in f:
    words=line[:-1].split(' ')
    l = len(words)
    s=''
    for i in range(l):
        if words[i]!='<blank>':
            s = s+words[i]
            if i!=l-1:
                s = s+' '
    if i == l-1:
        s = s + '\n'
    o.write(s)
