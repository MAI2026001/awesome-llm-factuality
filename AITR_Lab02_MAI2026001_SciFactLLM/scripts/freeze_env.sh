#!/usr/bin/env bash
# Record the versions actually installed for the packages this project uses.
pip freeze | grep -i -E "^(numpy|pandas|scikit-learn|matplotlib|pyyaml|torch|transformers|accelerate|bitsandbytes|tokenizers)==" > requirements.lock.txt
python --version > results/python_version.txt
echo "wrote requirements.lock.txt"
