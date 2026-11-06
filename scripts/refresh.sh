set -e
python src/download_data.py
python src/clean_data.py
python src/make_pending.py
python src/predict.py
python src/snapshot.py
python src/simulate.py