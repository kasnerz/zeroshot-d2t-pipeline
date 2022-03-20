#!/bin/bash
set -e

mkdir -p "data/d2t"
cd "data/d2t"

# WebNLG
if [[ ! -d webnlg ]]; then
    echo "======================================================"
    echo "Downloading the WebNLG data..."
    echo "======================================================"
    git clone "https://github.com/ThiagoCF05/webnlg.git"
    cd "webnlg"
    git checkout "12ca34880b225ebd1eb9db07c64e8dd76f7e5784" 2>/dev/null
    cd ..
fi

# Cleaned E2E
if [[ ! -d e2e ]]; then
    echo "======================================================"
    echo "Downloading the E2E data..."
    echo "======================================================"
    git clone "https://github.com/tuetschek/e2e-cleaning.git"
    cd "e2e-cleaning"
    git checkout "3cf74701a07a620b36bb63a6b771f02d9c1315a3" 2>/dev/null
    mv "cleaned-data/test-fixed.csv" "cleaned-data/test.csv"
    mv "cleaned-data/train-fixed.no-ol.csv" "cleaned-data/train.csv"
    mv "cleaned-data/devel-fixed.no-ol.csv" "cleaned-data/dev.csv"
    cd ..

    mv "e2e-cleaning" "e2e"
fi

cd ../..

# E2E metrics
if [[ ! -d e2e-metrics ]]; then
    echo "======================================================"
    echo "Cloning the E2E metrics..."
    echo "======================================================"
    git clone "https://github.com/tuetschek/e2e-metrics.git"
    mv "e2e-metrics" "e2e_metrics"
    cd "e2e_metrics"
    git checkout "dca5d301a97f7264b0827fb5589c0cc51008b5d7" 2>/dev/null
    cd ..
fi