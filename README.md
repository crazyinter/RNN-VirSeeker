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
