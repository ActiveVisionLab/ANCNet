pip install -r requirements.txt
wget -O ancnet.zip https://www.dropbox.com/s/bjul4f5z7beq3um/ancnet.zip?dl=0 
unzip -q ancnet.zip 
rm ancnet.zip

python eval_pf_pascal.py --a 0.1 --num_examples 5 

