import os
import gzip
import wget

def main():
    if not os.path.exists('data'):
        print("Creating a data directory")
        os.mkdir('data')
    
    if not os.path.exists('data/GoogleNews-vectors-negative300.bin.gz'):
        url = 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'
        wget.download(url, out = "./data")
        f_in = gzip.open('../data/GoogleNews-vectors-negative300.bin.gz', 'rb')
        f_out = open('../data/GoogleNews-vectors-negative300.bin', 'wb')
        f_out.writelines(f_in)

