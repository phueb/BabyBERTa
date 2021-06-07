


for file in data/* ; do
    for model in babyberta_ao-childes babyberta_ao-newsela babyberta_wikipedia-1; do
        echo "Scoring pairs in ${file}..."
        mkdir -p output/${model}/
        python3 main.py --mode ref --model ${model} --gpus 0 --split-size 500 ${file} > output/${model}/$(basename ${file} .txt).lm.json
    done
done