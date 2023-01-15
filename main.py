## importing the tokenizer and subword BPE trainer
from tokenize import tokenize

from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import BpeTrainer, WordLevelTrainer, \
    WordPieceTrainer, UnigramTrainer
## a pretokenizer to segment the text into words
from tokenizers.pre_tokenizers import Whitespace

unk_token = "<UNK>"  # token for unknown words
spl_tokens = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]  # special tokens


def prepare_tokenizer_trainer(alg):
    """
    Prepares the tokenizer and trainer with unknown & special tokens.
    """
    if alg == 'BPE':
        tokenizer = Tokenizer(BPE(unk_token=unk_token))
        trainer = BpeTrainer(special_tokens=spl_tokens)
    elif alg == 'UNI':
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(unk_token=unk_token, special_tokens=spl_tokens)
    elif alg == 'WPC':
        tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
        trainer = WordPieceTrainer(special_tokens=spl_tokens)
    else:
        tokenizer = Tokenizer(WordLevel(unk_token=unk_token))
        trainer = WordLevelTrainer(special_tokens=spl_tokens)

    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer, trainer


def train_tokenizer(files, alg='WPC'):
    """
    Takes the files and trains the tokenizer.
    """
    tokenizer, trainer = prepare_tokenizer_trainer(alg)
    tokenizer.train(files, trainer)  # training the tokenzier
    tokenizer.save(f"./{alg}-tokenizer-trained.json")
    tokenizer = Tokenizer.from_file(f"./{alg}-tokenizer-trained.json")
    return tokenizer


def read_txt(file_name):
    # Using readlines()
    file1 = open(file_name, 'r', encoding="utf8")
    Lines = file1.readlines()
    sents = []

    # Strips the newline character
    for line in Lines:
        sents.append(line.strip())
    return sents


def write_txt(sents, output_file):
    with open(output_file, 'w', encoding="utf8") as f:
        for line in sents:
            f.write(f"{line}\n")


for alg in ['BPE', 'UNI', 'WPC']:
    print("----", alg, "----")
    trained_tokenizer = train_tokenizer(["tr_boun-ud-train.txt"], alg)

    for split in ["train", "test"]:
        the_sents = read_txt(f"tr_boun-ud-{split}.txt")
        tokenized_sents = []
        for input_string in the_sents:
            output = trained_tokenizer.encode(input_string)
            tokenized_sents.append(" ".join(output.tokens))

        write_txt(tokenized_sents, f"{alg}_{split}.txt")

