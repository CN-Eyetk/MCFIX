#!/bin/bash
filepath="data/eyetrack"
langs=("cantonese" "mandarin")
result_path="result/regression/res.txt"
mappers="brr elast gbdt lgb lr mlp plsr rf rr"

for file in $(ls data/eyetrack);do
    echo "$file"
    python3 utils/addsp.py -i "data/eyetrack/$file" \
                           -o "data/processed/$file" \
                           -m 'uer/gpt2-chinese-cluecorpussmall' \
                           -n 'clue' 
done

for file in $(ls data/processed);do
    echo "$file"
    python3 utils/addsp.py -i "data/processed/$file" \
                            -o "data/processed/$file" \
                            -m 'jed351/gpt2_tiny_zh-hk-wiki' \
                            -n 'jed351' 
done

for file in $(ls data/processed);do
    echo "$file"
    python3 utils/preprocessing.py -i "data/processed/$file" \
                            -o "data/processed/$file "
done

for lang in "${langs[@]}"; do
    if [[ $lang == "cantonese" ]]
    then
        model_names=("clue" "jed351")
        models=("uer/gpt2-chinese-cluecorpussmall" 'jed351/gpt2_tiny_zh-hk-wiki')
    else
        model_names=("clue")
        models=("uer/gpt2-chinese-cluecorpussmall")
    fi
    for i in "${!model_names[@]}";do
        for usepoly in {0..1};do
            for usegpt in {0..1};do
                echo "Current Language Variety:$lang "
                echo "Using Polynomial:$usepoly "
                echo "Using Gpt embedding:$usegpt "
                echo "Gpt Suprisal: ${model_names[i]}"
                if [[ $usegpt == 1 ]];then
                echo "Gpt Embedding from: ${models[i]}"
                fi
                python3 main.py -i "$filepath"\
                        -l "$lang" \
                        -mn "${model_names[i]}" \
                        -m "${models[i]}" \
                        -g "$usegpt" \
                        -p "$usepoly" \
                        -mp "$mappers" \
                        -r "$result_path"
            done
        done
    done
done