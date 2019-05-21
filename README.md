# RNN-VirSeeker
Version 1.1 <br>
Authors: Yan Miao, Fu Liu, Yun Liu <br>
Maintainer: Yan Miao miaoyan17@mails.jlu.edu.cn 

# Description
This package provides a deep learning method for identification of viral contigs from metagenomic data in a fasta file. The method has the ability to identify viral contigs with short length (<500bp) from metagenomic data.

The prediction model is a deep learning based Recurrent Neural Network (RNN) that learns the high-level features of each contig to distinguish virus from host sequences. The model was trained using equal number of known viral and host sequences from NCBI RefSeq database. Before training, those known sequences were firstly split into a number of non-overlapping contigs with a length of 500bp and then were encoded by transforming (A, T, G, C) to (1, 2, 3, 4), respectively.  For a query sequence shorter than 500bp, it should be first zero-padded up to 500bp. Then the sequence is predicted by the RNN model trained with previously known sequences.

# Dependencies
To utilize RNN-VirSeeker, Python packages "sklearn", "numpy" and "matplotlib" are needed to be previously installed.

In convenience, download Anaconda from https://repo.anaconda.com/archive/, which contains most of needed packages.

To insatll tensorflow, start "cmd.exe" and enter <br>
```
pip install tensorflow
```
Our codes were all edited by Python 3.6.5 with TensorFlow 1.3.0.
# Usage
It is simple to use RNN-VirSeeker for users' database. <br>
There are two ways for users to train the model using `train.py`.
* Using our original training database (containing 4500 viral sequences and 4500 host sequences of length 500bp) `"rnn_train.csv"`. <br>
Users can utilize the trained model directly to test query contigs. Or you can make some changes to the hyperparameters, and then retrain the model.
* Using users' own database in a ".csv" format. <br>
	* Firstly, chose a set of hyperparameters to train your dataset.
	* Secondly, train and refine your model using your dataset according to the performance on a related validation dataset.
	* Finally, utilize the saved well trained model to identify query contigs. 
Note: Before training, set the path to where the database is located. All labels should be encoded to one-hot labels.

To make a prediction, users' own query contigs should be edited into a ".csv" file, where every line contains a single query contig. Through `test.py`, RNN-VirSeeker will give a set of scores to each query contig, higher of which represents its classification result.

# Copyright and License Information
Copyright (C) 2019 Jilin University

Authors: Yan Miao, Fu Liu, Yun Liu

This program is freely available as Python at https://github.com/crazyinter/RNN-VirSeeker.

Commercial users should contact Mr. Miao at miaoyan17@mails.jlu.edu.cn, copyright at Jilin University.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
