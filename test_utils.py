from run_muse import *
from src.utils import load_embeddings, normalize_embeddings

import torch.nn as nn

DATA_DIR = "data"

def load_model(src_lang, tgt_lang, model_path):
    params = argparse.Namespace()
    params.emb_dim = 300
    params.max_vocab = 200000
    params.cuda = True
    params.src_lang = src_lang
    params.tgt_lang = tgt_lang
    params.src_emb = os.path.join(DATA_DIR, "wiki.%s.vec" % params.src_lang)
    params.tgt_emb = os.path.join(DATA_DIR, "wiki.%s.vec" % params.tgt_lang)
    params.dico_eval = os.path.join(DATA_DIR, "%s-%s.5000-6500.txt" % (params.src_lang, params.tgt_lang))
    params.normalize_embeddings = ""

    src_emb, tgt_emb, mapping, discriminator = build_model(params, True)
    trainer = Trainer(src_emb, tgt_emb, mapping, discriminator, params)

    #load mapping
    to_reload = torch.from_numpy(torch.load(model_path))
    W = trainer.mapping.weight.data
    assert to_reload.size() == W.size()
    W.copy_(to_reload.type_as(W))

    evaluator = Evaluator(trainer)

    return params, trainer, evaluator

