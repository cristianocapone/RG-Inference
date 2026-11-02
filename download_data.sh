#!/usr/bin/env bash
# Download the public data sets used in the paper into data/.
#   data/lonardoni2017_spont/      spontaneous activity, rat hippocampal cultures, 3Brain 4096 HD-MEA
#   data/lonardoni2017_drug/       same cultures, basal vs bicuculline (BIC/) and APV (APV/)
#       Lonardoni et al. (2017) PLoS Comput Biol 13:e1005672; Zenodo record 4977727; CC0.
#   data/stringer2019_neuropixels/ eight-probe Neuropixels, 3 mice, spontaneous behaviour
#       Stringer et al. (2019) Science 364:eaav7893; figshare 10.25378/janelia.7739750; CC BY-NC 4.0.
# About 900 MB in total.
set -euo pipefail
cd "$(dirname "$0")"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

zen="https://zenodo.org/api/records/4977727/files"
fig="https://ndownloader.figshare.com/files"

fetch_zip () {   # url, destination folder
    local url=$1 dest=$2
    mkdir -p "$dest" "$TMP/x"
    curl -L --fail -o "$TMP/a.zip" "$url"
    rm -rf "$TMP/x"/*
    unzip -q -o "$TMP/a.zip" -d "$TMP/x"
    echo "$TMP/x"
}

# --- Lonardoni 2017, spontaneous ---
if [ ! -f data/lonardoni2017_spont/d3_spont.mat ]; then
    x=$(fetch_zip "$zen/SPONT_recordings.zip/content" data/lonardoni2017_spont)
    find "$x" -name '*_spont.mat' -exec mv {} data/lonardoni2017_spont/ \;
fi

# --- Lonardoni 2017, pharmacology ---
for cond in BIC APV; do
    if [ ! -d "data/lonardoni2017_drug/$cond" ] || [ -z "$(ls -A "data/lonardoni2017_drug/$cond" 2>/dev/null)" ]; then
        mkdir -p "data/lonardoni2017_drug/$cond"
        x=$(fetch_zip "$zen/${cond}_recordings.zip/content" "data/lonardoni2017_drug/$cond")
        find "$x" -name '*.mat' -exec mv {} "data/lonardoni2017_drug/$cond/" \;
    fi
done

# --- Stringer 2019, Neuropixels ---
S=data/stringer2019_neuropixels
mkdir -p "$S"
[ -f "$S/ephysLoad.m" ]        || curl -L --fail -o "$S/ephysLoad.m" "$fig/14570105"
[ -f "$S/probeBorders.mat" ]   || curl -L --fail -o "$S/probeBorders.mat" "$fig/14653298"
[ -f "$S/probeLocations.mat" ] || curl -L --fail -o "$S/probeLocations.mat" "$fig/14570108"
if [ ! -f "$S/spksKrebs_Feb18.mat" ]; then
    x=$(fetch_zip "$fig/14405057" "$S")
    find "$x" -name 'spks*_Feb18.mat' -exec mv {} "$S/" \;
fi

echo "done:"
du -sh data/*
