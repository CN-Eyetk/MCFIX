from textutils import *
from wsurprisal import *
import pandas as pd
import argparse
from transformers import AutoTokenizer
parser = argparse.ArgumentParser()
parser.add_argument("-i","--input",type=str)
parser.add_argument("-o","--output",type=str)
parser.add_argument("-m","--model",type=str)
parser.add_argument("-n","--modelname",type=str)
args = parser.parse_args()


def parse_surprisal(df,scorer):
    df["new_text"] = df["IA_LABEL"].apply(lambda x:text_normalizer(x))
    sent_ids = pd.unique(df["SENTENCE"])
    stimulis = []
    for sid in sent_ids:
        frame = df[df["SENTENCE"] == sid]
        stimuli = frame["new_text"].values.tolist()
        stimulis.append(stimuli)
    print(f"stimulis:{stimulis[0:3]}")
    scorer.load_stimuli(stimulis)
    word_score_by_sent = scorer.get_word_score()
    tokens = [x[0] for y in word_score_by_sent for x in y]
    words = df["new_text"]
    if "Unnamed: 0" in df.columns:
        del df["Unnamed: 0"]
    #del df["new_text"]
    for w,t in zip(words, tokens):
        if w != t:
            print(f"it is noticed that {w} is not identical to {t}")
    del df["new_text"]
    df[f"Sp_{args.modelname}"] = [x[1] for y in word_score_by_sent for x in y]
    return word_score_by_sent, df
        
if __name__ == "__main__":
    filename = args.input
    df = load_dep(filename)
    scorer = WordSurprisal(args.model,"cuda")
    res = parse_surprisal(df,scorer)
    df.to_csv(args.output ,index=None, sep="\t")

#def nothing():
#    stimulis = "这*是*很久*之前*的*事情*了。/小明*和*小红*难道*不*清楚*吗？"
#    stimulis = stimulis.split("/")
#    stimulis = [x.split("*") for x in stimulis]
#    scorer.load_stimuli(stimulis)
#    char_scores = scorer.get_char_score()
#    print("char scores:")
#    print(char_scores)
#    print("word scores:")
#    word_score_by_sent = scorer.get_word_score()
#    print(word_score_by_sent)
