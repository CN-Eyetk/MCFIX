import re
import pandas as pd
from minicons import scorer
from transformers import AutoTokenizer
from transformers import GPT2Tokenizer, GPT2Model, pipeline
import torch

def alignchar2word(chars, words):
    labels = []
    chars = [x.strip() for x in chars]
    cur_state = ""
    cur_point = 0
    for i, char in enumerate(chars):
        char = char.replace("#","")
        if char == "[UNK]" or char == "<unk>":
            char = words[cur_point][len(cur_state)]
            #print(f"{words[cur_point][len(cur_state)]} is UNK")
        cur_state += char
        #print(cur_state)
        #print(words[cur_point][:len(cur_state)])
        assert words[cur_point][:len(cur_state)] == cur_state
        labels.append(cur_point)
        if len(words[cur_point]) == len(cur_state):
            cur_state = ""
            cur_point += 1
    return labels


def text_normalizer(text):
    text = text.replace("⋯",'⋯⋯')
    text = text.replace("…",'⋯⋯')
    text = text.replace(" ","")
    text = text.replace("‘","'")
    text = text.replace("’","'")
    text = text.replace("“","\"")
    text = text.replace("”","\"")
    text = text.replace("B612","b612")
    text = text.replace("—","﹣")
    text = text.replace("噉","咁")
    text = text.replace("\u200b","")
    text = re.compile("^[。」]+(?=[\w\u4e00-\u9fa5])").sub("",text)
    return text

def text_normalizer_bert(text):
    text = text.replace("，",',')
    text = text.replace("：",':')
    text = text.replace("⋯⋯","⋯")
    text = text.replace("！","!")
    text = text.replace("“","\"")
    text = text.replace("”","\"")
    text = text.replace("B612","b612")
    text = text.replace("—","﹣")
    text = text.replace("噉","咁")
    return text


def load_corpus(path):
    data = open(path,"r").read()
    data = data.split("\n")
    data = [x for x in data if len(x)>0]
    return data

def load_dataset(path,tokenizer,block_size):
    dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=path,
          block_size=block_size)
     
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return dataset,data_collator


def load_dep(filename):
    df = pd.read_table(filename)
    #print(df.head())
    return df

def load_eyetrack(filename):
    df = pd.read_excel(filename,engine='openpyxl')
    #print(df.head())
    return df

def parse_vec(df,scorer,model_name):
    df["new_text"] = df["IA_LABEL"].apply(lambda x:text_normalizer(x))
    sent_ids = pd.unique(df["SENTENCE"])
    stimulis = []
    for sid in sent_ids:
        frame = df[df["SENTENCE"] == sid]
        stimuli = frame["new_text"].values.tolist()
        stimulis.append(stimuli)
    print(f"stimulis:{stimulis[0:3]}")
    scorer.load_stimuli(stimulis)
    word_vec_by_sent = scorer.get_word_vecs()
    tokens = [x[0] for y in word_vec_by_sent for x in y]
    words = df["new_text"]
    if "Unnamed: 0" in df.columns:
        del df["Unnamed: 0"]
    for w,t in zip(words, tokens):
        if w != t:
            print(f"it is noticed that {w} is not identical to {t}")
    new_df = {}
    vec_size = len(word_vec_by_sent[0][0][1])
    for i in range(vec_size):
        new_df[f"Vecs_{model_name}_{i}"] = [x[1][i] for y in word_vec_by_sent for x in y]
    new_df = pd.DataFrame(new_df)
    new_df = pd.concat([df, new_df], axis = 1)
    return word_vec_by_sent, new_df

class WordSurprisal:
    def __init__(self, model_name, device):
        if "gpt" in model_name:
            self.model = scorer.IncrementalLMScorer(model_name, device)
            self.char2vec = pipeline(model = model_name, task = "feature-extraction", return_tensors = False, deivce = 0)
        elif "bert" in model_name:
            self.model = scorer.MaskedLMScorer(model_name, device)
            self.char2vec = pipeline(model=model_name, task="feature-extraction", return_tensors = False, deivce = 0)
        elif "cino" in model_name:
            self.model = scorer.MaskedLMScorer(model_name, device)
            self.model.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
            self.char2vec = pipeline(model=model_name, task="feature-extraction", return_tensors = False, deivce = 0)
        else:
            self.model = scorer.IncrementalLMScorer(model_name, device)
            self.char2vec = pipeline(model=model_name, task="feature-extraction", return_tensors = False, deivce = 0)
        self.model_name = model_name
        self.device = device
        self.stimuli = None
    def load_stimuli(self, stimuli_tokens): #[[w1, w2],[w1,w2,w3]...]
        self.stimuli_tokens = stimuli_tokens
        self.stimuli = ["".join(x for x in y) for y in stimuli_tokens] # [[w1, w2],[w1,w2,w3]...] -> [w1w2, w1w2w3]
        #print("Stimulis:")
        #for i in range(len(self.stimuli_tokens)):
        #    print(self.stimuli_tokens[i])
        #    print(self.stimuli[i])
        #    print("===========================")
    def get_char_score(self):
        assert self.stimuli != None
        scores = []
        n = len(self.stimuli)
        for i in range(n):
            score = self.model.token_score([self.stimuli[i]], surprisal = True, base_two = True)
            scores += score
        if "cino" not in self.model_name and "bert" not in self.model_name:
            scores = [x[1:-1] for x in scores] 
        else:
            scores = [[y for y in x if len(y[0])>0] for x in scores]
        return scores
    def get_char_vecs(self):
        assert self.stimuli != None
        vecs = []
        n = len(self.stimuli)
        for i in range(n):
            cur_stimuli = self.stimuli[i]
            tokens = self.model.tokenizer.tokenize(cur_stimuli)
            vec = self.char2vec(cur_stimuli)[0]
            vec = vec[1:-1]
            token_vec = [(t,v) for t,v in zip(tokens, vec)]
            vecs.append(token_vec)
        return vecs     #vecs = [[(c1,vec1),(c2,vec2),(c3,vec3)],[(c1,vec1),(c2,vec2)]]
    def get_word_score(self):
        char_score = self.get_char_score()
        word_score_by_sent = []
        for wseq,y in zip(self.stimuli_tokens, char_score):
            chars = [c for c,_ in y]
            scores = [s for _,s in y]
            try:
                word_label_for_char = alignchar2word(chars, wseq)
                word_scores = []
                for m in range(max(word_label_for_char)+1):
                    word_scores.append(sum([s for i, s in enumerate(scores) if word_label_for_char[i] == m]))
                word_score_by_sent.append([(w,s) for w,s in zip(wseq, word_scores)])
            except:
                print("Error in Align")
                print(f"words = {wseq}")
                print(f"chars = {chars}")
                print("=============")
        return word_score_by_sent
    def get_word_vecs(self):
        char_vecs = self.get_char_vecs()
        word_vecs_by_sent = []
        for wseq,y in zip(self.stimuli_tokens, char_vecs):
            chars = [c for c,_ in y]
            scores = [s for _,s in y]
            try:
                word_label_for_char = alignchar2word(chars, wseq)
            except:
                print("Error in Align")
                print(f"words = {wseq}")
                print(f"chars = {chars}")
                print("=============")
            word_vecs = []
            for m in range(max(word_label_for_char)+1):
                word_vec = [s for i, s in enumerate(scores) if word_label_for_char[i] == m]
                word_vec = torch.tensor(word_vec)
                #print(f"prepooling-{word_vec.size()}")
                word_vec = torch.mean(word_vec, dim = 0)
                #print(f"postpooling-{word_vec.size()}")
                word_vecs.append(word_vec.tolist())
            word_vecs_by_sent.append([(w,s) for w,s in zip(wseq, word_vecs)])
        return word_vecs_by_sent

def merge_eyetrack(files,path):
    dfs = []
    for file in files:
        if "simp" in file:
            lang = 0
        elif "trad" in file:
            lang = 1
        if "NR" in file:
            task = "NR"
        elif "TSR" in file:
            task = "TSR"
        if "first" in file:
            qtype = "first"
        elif "second" in file:
            qtype = "second"
        df_cur = load_dep(f"{path}/{file}")
        df_cur["TASK"] = task
        df_cur["QTYPE"] = qtype
        df_cur["LANG"] = lang
        dfs.append(df_cur)
    df = pd.concat(dfs, axis = 0)
    print(f"Shape of Whole Data={df.shape}")
    return df

def merge_eyetrack_with_subj(files,path):
    dfs = []
    for file in files:
        if "simp" in file:
            lang = 0
        elif "trad" in file:
            lang = 1
        if "NR" in file:
            task = "NR"
        elif "TSR" in file:
            task = "TSR"
        if "first" in file:
            qtype = "first"
        elif "second" in file:
            qtype = "second"
        df_cur = load_eyetrack(f"{path}/{file}")
        df_cur["TASK"] = task
        df_cur["QTYPE"] = qtype
        df_cur["LANG"] = lang
        dfs.append(df_cur)
    df = pd.concat(dfs, axis = 0)
    print(f"Shape of Whole Data={df.shape}")
    return df 


def orth_2_n_syl(word):
    pattern = re.compile(r"[B0-9１２\u4E00-\u9FFF]")
    n_syl = len(pattern.findall(word))
    return n_syl

def load_dict(path):
    pd_dic = pd.read_table(path, sep = "\t")
    return pd_dic

def get_canto_freq(word,pd_dic):
    has_word = re.compile(r"[B0-9１２\u4E00-\u9FFF]+").search(word)
    if has_word:
        word = has_word.group()
        targ = pd_dic[pd_dic["Word"] == word].index
        if len(targ) > 0:
            freq = pd_dic.loc[targ,'Written (per million tokens)'].item()
        else:
            freq = 0
    else:
        freq = 0
    return freq

def get_mand_freq(word,pd_dic):
    has_word = re.compile(r"[B0-9１２\u4E00-\u9FFF]+").search(word)
    if has_word:
        word = has_word.group()
        targ = pd_dic[pd_dic["词语"] == word].index
        if len(targ) > 0:
            freq = pd_dic.loc[targ,'频率（%）'].item() / 100 
            freq = freq * 1000000
        else:
            freq = 0
    else:
        freq = 0
    return freq

def gen_pre_punct(word):
    pre_punct = re.compile(r"^[^B0-9１２\u4E00-\u9FFF]+[B0-9１２\u4E00-\u9FFF]+")
    if pre_punct.search(word):
        return True
    else:
        return False

def gen_post_punct(word):
    #pre_punct = re.compile(r"^[^B0-9１２\u4E00-\u9FFF]+[B0-9１２\u4E00-\u9FFF]+")
    post_punct = re.compile(r"[B0-9１２\u4E00-\u9FFF]+[^B0-9１２\u4E00-\u9FFF]+$")
    if post_punct.search(word):
        return True
    else:
        return False

def gen_word_pos(df, norm=True):
    n_row = df.shape[0]
    df["Word_Position"] = n_row * [0]
    sent_lengths = []
    for i in range(0,n_row):
        cur_sent = df.loc[i,"SENTENCE"].item()
        if norm:
            cur_sent_length = df[df["SENTENCE"] == cur_sent].shape[0]
            sent_lengths.append(cur_sent_length)
        if i == 0:
            cur_pos = 1
            df.loc[i,"Word_Position"] = cur_pos
        else:
            last_word_sent = df.loc[i-1,"SENTENCE"].item()
            if cur_sent == last_word_sent:
                cur_pos = df.loc[i-1,"Word_Position"].item() + 1
            else:
                cur_pos = 1
            df.loc[i,"Word_Position"] = cur_pos
    if norm:
        df["Lengths"] = sent_lengths
        df["Word_Position"] = df["Word_Position"] / df["Lengths"]
    return df

def gen_prev_freq(df):
    n_row = df.shape[0]
    df["Prev_Freq"] = [-1] + (n_row-1) * [0]
    for i in range(1,n_row):
        cur_sent = df.loc[i,"SENTENCE"].item()
        last_word_sent = df.loc[i-1,"SENTENCE"].item()
        if cur_sent == last_word_sent:
            value = df.loc[i-1,"Freq"].item()
        else:
            value = -1
        df.loc[i,"Prev_Freq"] = value
    return df

def gen_prev_n_syl(df):
    n_row = df.shape[0]
    df["Prev_N_SYL"] = [-1] + (n_row-1) * [0]
    for i in range(1,n_row):
        cur_sent = df.loc[i,"SENTENCE"].item()
        last_word_sent = df.loc[i-1,"SENTENCE"].item()
        if cur_sent == last_word_sent:
            value = df.loc[i-1,"N_SYL"].item()
        else:
            value = -1
        df.loc[i,"Prev_N_SYL"] = value
    return df

def gen_prev_pos(df):
    n_row = df.shape[0]
    df["Prev_POS"] = ["PUNCT"] + (n_row-1) * [0]
    for i in range(1,n_row):
        cur_sent = df.loc[i,"SENTENCE"].item()
        last_word_sent = df.loc[i-1,"SENTENCE"].item()
        if cur_sent == last_word_sent:
            value = df.loc[i-1,"upos*"]
        else:
            value = "PUNCT"
        df.loc[i,"Prev_POS"] = value
    return df

def gen_prev_surprisal(df):
    n_row = df.shape[0]
    df["Prev_Suprisal"] = [0] + (n_row-1) * [0]
    for i in range(1,n_row):
        cur_sent = df.loc[i,"SENTENCE"].item()
        last_word_sent = df.loc[i-1,"SENTENCE"].item()
        if cur_sent == last_word_sent:
            value = df.loc[i-1,"upos*"]
        else:
            value = "PUNCT"
        df.loc[i,"Prev_POS"] = value
    return df

