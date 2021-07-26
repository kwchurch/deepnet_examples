
download this to trans.gz
https://drive.google.com/file/d/1p5HoOJ_HJStA4b9uzNmdRkLgdijsMgjO/view?usp=sharing

# ../near.py (and other programs) will be faster if you specify filenames ending in .annoy
# you can create them with the following:

python save_embeddings.py trans.gz

python ../near.py --list 1 |
while read f
do
echo working on $f
python save_embedding.py $f
done

