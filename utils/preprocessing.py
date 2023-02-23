import pandas as pd
import os
import re
from textutils import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i","--input",type=str)
parser.add_argument("-l","--lang",type=str)
parser.add_argument("-o","--output",type=str)
parser.add_argument("-p","--addprev",type=int,default = 1)
args = parser.parse_args()


if __name__ == "__main__":
    filename = args.input
    lang = args.lang
    df = load_dep(filename)
    df = gen_word_pos_new(df) #Add WORD Position
    if args.addprev > 0:
        #df = gen_word_pos(df) 
        df = gen_prev_freq(df) #Add Previous Frequency
        df = gen_prev_n_syl(df) #Add Previous N Syllable
    df.to_csv(args.output ,index=None, sep="\t")