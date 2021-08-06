import sys

col1 = []
col2 = []
scores = []

left_context = 5
right_context = 5

def output_line(p):
    try:
        print('%s\t%s\t%50s\t%s' % (scores[p], col2[p], ' '.join(col1[p-left_context:p]), ' '.join(col1[p:p+right_context])))
    except:
        print('%s\t%s\t*** ERROR ***' % (scores[p], col1[p]))        

for line in sys.stdin:
    fields = line.rstrip().split()
    if len(fields) < 2: continue

    col2_fields = fields[1].split('|')
    if len(col2_fields) == 2:
        col1.append(fields[0])
        tok,score = col2_fields[0:2]
        col2.append(tok)
        scores.append(float(score))

for i,x in enumerate(col1):
    if x == sys.argv[1]:
        output_line(i)
    
