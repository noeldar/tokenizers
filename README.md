# tokenizers

# File Contents

* main.py : trains tokenizers with HuggingFace library.

**Inputs:<br />
  -tr_boun-ud-train.txt: train corpus<br />
  -tr_boun-ud-test.txt: eval corpus<br />

**Outputs:<br />
  -UNI/WPC/BPE_train.txt: train corpus tokenizations<br />
  -UNI/WPC/BPE_test.txt: eval corpus tokenizations<br />
  -UNI/WPC/BPE-tokenizer-trained.json: trained tokenizers files<br />
  
  
  * preprocess.py/language_model.py : Bigram language models training codes<br />

**Inputs:<br />
  -UNI/WPC/BPE_train.txt: train corpus tokenizations<br />
  -UNI/WPC/BPE_test.txt: eval corpus tokenizations<br />

**Outputs:<br />
  perplexity scores for test.txts<br />
  
  
