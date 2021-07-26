import gzip,gensim,os,sys,gensim.downloader
from gensim.models import KeyedVectors

for model_string in sys.argv[1:]:
    if model_string == 'trans.gz':
        with open('trans.txt', 'w') as outf:
            with gzip.open('trans.gz', 'r') as fd:
                lines = fd.read().decode('utf-8').split('\n')
                if len(lines[-1]) == 0:
                    lines = lines[0:-1]
                print(str(len(lines)) + ' ' + str(len(lines[0].split())-1), file=outf)
                print('\n'.join(lines), file=outf)
        M = KeyedVectors.load_word2vec_format('trans.txt', binary=False)
        M.save('trans.annoy')
    else:
        M = gensim.downloader.load(model_string)
        M.save(model_string + '.annoy')
        

        
    
