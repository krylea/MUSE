from run_muse import joint_procrustes

import argparse
import os

DATA_DIR = "data"
MODEL_DIR = "dumped"
OUTPUT_DIR = "results"

def parse_args():
    parser = argparse.ArgumentParser(description='Unsupervised training')

    parser.add_argument("--src_lang", type=str, default='en', help="Source language")
    parser.add_argument("--tgt_langs", type=str, default='es', help="Target languages", nargs='+')
    parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")

    return parser.parse_args()


def load_models(src_lang, tgt_lang, n_exp=5):
    s2t_exp_dir = os.path.join(MODEL_DIR, "%s%s" % (src_lang, tgt_lang))
    t2s_exp_dir = os.path.join(MODEL_DIR, "%s%s" % (src_lang, tgt_lang))

    s2t_models = []
    t2s_models = []
    for i in range(n_exp):
        s2t_models.append(os.path.join(s2t_exp_dir, str(i), "best_mapping.pth"))
        t2s_models.append(os.path.join(t2s_exp_dir, str(i), "best_mapping.pth"))

    return s2t_models, t2s_models

