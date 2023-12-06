py -m pip --version
pip install -r requirements.txt
#tar -xf oxford-iiit-pet/images.tar -C oxford-iiit-pet/
py create_train_test_split.py
py tests/testCUDA.py
