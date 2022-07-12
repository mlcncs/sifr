
# Requirements:

   ubuntu os
   
   python3
   
   g++
   
   armadillo matrix library

# armadillo install
1. run the following command to install `OpenBLAS` and `LAPACK`
   ```bash
   sudo apt-get install libopenblas-dev
   sudo apt-get install liblapack-dev
   sudo apt-get install libarpack2-dev
   sudo apt-get install libsuperlu-dev
   ```
2. download and unzip files
   ```bash
   wget https://nchc.dl.sourceforge.net/project/arma/armadillo-11.2.3.tar.xz
   tar -xvJf  armadillo-11.2.3.tar.xz
   ```
3. install  armadillo
   ``` bash
   cd armadillo-11.2.3
   cmake .
   make
   sudo make install
   ```

# Run

1.  download the dataset into `./data/or`:  'a9a','w8a','ijcnn1','SUSY','HIGGS'
   ```bash
   wget -P ./data/or https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a

   ```
3.  Preprocessed data dataset
   ```bash
   
   mkdir -p ./data/after
   python split_dataset.py
   ```
3. build run file
   ```bash
   g++ main.cpp -o main -std=c++11 -O2 -larmadillo
   ```
4.  run 
   ```bash
   mkdir  sqhinge
   python exp.py
   ```
