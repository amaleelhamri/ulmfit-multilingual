#!/usr/bin/env bash
# Script to download a Wikipedia dump

# Script is partially based on https://github.com/facebookresearch/fastText/blob/master/get-wikimedia.sh
ROOT="data"
echo "Saving data in ""$ROOT"

if [ "$1" == "" ] ; then
    read -r -p "Choose a language (e.g. en, bh, fr, etc.): " choice
    LANG="$choice"
else
    LANG="$1"
fi
echo "Chosen language: ""$LANG"

DUMP_DIR="${ROOT}/wiki_dumps"
EXTR_DIR="${ROOT}/wiki_extr"
WIKI_DIR="${ROOT}/wiki"
EXTR="wikiextractor"
mkdir -p "${ROOT}"
mkdir -p "${DUMP_DIR}"
mkdir -p "${EXTR_DIR}"
mkdir -p "${WIKI_DIR}"

DUMP_FILE="${LANG}wiki-latest-pages-articles.xml.bz2"
DUMP_PATH="${DUMP_DIR}/${DUMP_FILE}"

if [ ! -f "${DUMP_PATH}" ]; then
  wget -c "https://dumps.wikimedia.org/""${LANG}""wiki/latest/""${DUMP_FILE}""" -P "${DUMP_DIR}"
else
  echo "${DUMP_PATH} already exists. Skipping download."
fi

# Check if directory exists
if [ ! -d "${EXTR}" ]; then
  git clone https://github.com/attardi/wikiextractor.git
  cd "${EXTR}"
  python setup.py install
  cd ..
fi

EXTR_PATH="${EXTR_DIR}/${LANG}"
if [ ! -d "${EXTR_PATH}" ]; then
  python wikiextractor/WikiExtractor.py -s --json -o "${EXTR_PATH}" "${DUMP_PATH}"
else
  echo "${EXTR_PATH} already exists. Skipping extraction."
fi

# merge all articles into one file. this will be used to train sentencepiece model
OUT_PATH="${WIKI_DIR}/${LANG}"
read -r -p "Continue to merge Wikipedia articles (y/n)? " choice
case "$choice" in
y|Y ) echo "Merging articles from ${EXTR_PATH} to ${OUT_PATH}...";;
n|N ) echo "Exiting";exit 1;;
* ) echo "Invalid answer";exit 1;;
esac

if [ ! -f "${OUT_PATH}/all.csv" ]; then
  python -m ulmfit.merge_wiki -i "${EXTR_PATH}" -o "${OUT_PATH}" 
else
  echo "${OUT_PATH}/all.csv already exists. Skipping merge."
fi

# train sentencepiece model from all merged text 
# and split data into train and valid for 3 different token sizes
python -m ulmfit.create_wikitext -i "${EXTR_PATH}"  -l "${LANG}" -o "${WIKI_DIR}" \
  --max_vocab 8000 --character_coverage 0.995 --input_sentence_size 1E7
