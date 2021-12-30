from datetime import datetime
import os
import argparse
import pytorch_lightning as pl
from problem import DataModule
from problem.ner_problem_train import NERProblemTrain
from problem.lit_recurrent_ner_train import LightningRecurrent_NERTrain
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

path = os.path.dirname(os.path.abspath(__file__))
today = datetime.today().strftime("%Y-%m-%d")
def input_chromosome(args):
    try:
        with open(args.checkpoint_file,'rb') as f:
            d = pickle.load(f)
            return np.array(d['population'][0])
    except:
        print('Read from pickle file')
    try:
        with open(args.file_name, 'rb') as f:
            d = pickle.load(f)
            return np.array(d['population'][0])
    except:
        with open(path + args.file_name, 'r') as f:
            chromosome = f.read()
    try:
        chromosome = chromosome.split()
        chromosome = [int(x) for x in chromosome]
        return np.array(chromosome)
    except:
        return np.array(chromosome)

def parse_args():
    parser = argparse.ArgumentParser()
    parser = NERProblemTrain.add_arguments(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DataModule.add_argparse_args(parser)
    parser = DataModule.add_cache_arguments(parser)
    parser.add_argument("--file_name", default= '/chromosome.txt', type=str)
    parser.add_argument("--checkpoint_file", default= '/checkpoint.pkl', type=str)
    parser.add_argument("--save_path", default = path + f"chromosome_trained_weights.gene_nas.{today}.pkl", type= str)
    parser = LightningRecurrent_NERTrain.add_model_specific_args(parser)
    parser = LightningRecurrent_NERTrain.add_learning_specific_args(parser)
    args = parser.parse_args()

    args.num_terminal = args.num_main + 1
    args.l_main = args.h_main * (args.max_arity - 1) + 1
    args.l_adf = args.h_adf * (args.max_arity - 1) + 1
    args.main_length = args.h_main + args.l_main
    args.adf_length = args.h_adf + args.l_adf
    args.chromosome_length = (
        args.num_main * args.main_length + args.num_adf * args.adf_length
    )
    args.D = args.chromosome_length
    args.mutation_rate = args.adf_length / args.chromosome_length

    return args

        
def main():
    args = parse_args()
    chromosome = input_chromosome(args)
    problem = NERProblemTrain(args= args)
    problem.evaluate(chromosome= chromosome)    

if __name__ == "__main__":
    main()