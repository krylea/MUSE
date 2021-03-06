import os
import time
import json
import argparse
from collections import OrderedDict
import numpy as np
import torch
import copy

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator
from src.evaluation.word_translation import get_word_translation_accuracy

VALIDATION_METRIC = 'mean_cosine-csls_knn_10-S2T-10000'
SAVE_DIR = "results"

def parse_args():
    parser = argparse.ArgumentParser(description='Unsupervised training')
    # our args
    parser.add_argument("--s2t_out", type=str, default=None, help="Output file")
    parser.add_argument("--t2s_out", type=str, default=None, help="Output file")
    parser.add_argument("--n_trials", type=int, default=5, help="number of runs")
    # main

    parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
    parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
    parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
    parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")
    # data
    parser.add_argument("--src_lang", type=str, default='en', help="Source language")
    parser.add_argument("--tgt_lang", type=str, default='es', help="Target language")
    parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
    parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
    # mapping
    parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the mapping as an identity matrix")
    parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")
    # discriminator
    parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
    parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimensions")
    parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
    parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="Discriminator input dropout")
    parser.add_argument("--dis_steps", type=int, default=5, help="Discriminator steps")
    parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
    parser.add_argument("--dis_most_frequent", type=int, default=75000, help="Select embeddings of the k most frequent words for discrimination (0 to disable)")
    parser.add_argument("--dis_smooth", type=float, default=0.1, help="Discriminator smooth predictions")
    parser.add_argument("--dis_clip_weights", type=float, default=0, help="Clip discriminator weights (0 to disable)")
    # training adversarial
    parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--epoch_size", type=int, default=1000000, help="Iterations per epoch")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
    parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")
    parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
    parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")
    # training refinement
    parser.add_argument("--n_refinement", type=int, default=5, help="Number of refinement iterations (0 to disable the refinement procedure)")
    # dictionary creation parameters (for refinement)
    parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
    parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
    parser.add_argument("--dico_build", type=str, default='S2T', help="S2T,T2S,S2T|T2S,S2T&T2S")
    parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
    parser.add_argument("--dico_max_rank", type=int, default=15000, help="Maximum dictionary words rank (0 to disable)")
    parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
    parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
    # reload pre-trained embeddings
    parser.add_argument("--src_emb", type=str, default="", help="Reload source embeddings")
    parser.add_argument("--tgt_emb", type=str, default="", help="Reload target embeddings")
    parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")

    # parse parameters
    params = parser.parse_args()


    # check parameters
    assert not params.cuda or torch.cuda.is_available()
    assert 0 <= params.dis_dropout < 1
    assert 0 <= params.dis_input_dropout < 1
    assert 0 <= params.dis_smooth < 0.5
    assert params.dis_lambda > 0 and params.dis_steps > 0
    assert 0 < params.lr_shrink <= 1
    #assert os.path.isfile(params.src_emb)
    #assert os.path.isfile(params.tgt_emb)
    #assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)
    assert params.export in ["", "txt", "pth"]

    return params

DATA_DIR = "data"
def set_default_args(params):
    params.src_emb = os.path.join(DATA_DIR, "wiki.%s.vec" % params.src_lang)
    params.tgt_emb = os.path.join(DATA_DIR, "wiki.%s.vec" % params.tgt_lang)
    params.dico_eval = os.path.join(DATA_DIR, "%s-%s.5000-6500.txt" % (params.src_lang, params.tgt_lang))
    params.out_file = os.path.join(SAVE_DIR, params.src_lang + params.tgt_lang + "_MUSE.txt")

    params2 = copy.deepcopy(params)
    params2.src_lang, params2.tgt_lang = params.tgt_lang, params.src_lang

    params2.src_emb = os.path.join(DATA_DIR, "wiki.%s.vec" % params2.src_lang)
    params2.tgt_emb = os.path.join(DATA_DIR, "wiki.%s.vec" % params2.tgt_lang)
    params2.dico_eval = os.path.join(DATA_DIR, "%s-%s.5000-6500.txt" % (params2.src_lang, params2.tgt_lang))
    params2.out_file = os.path.join(SAVE_DIR, params2.src_lang + params2.tgt_lang + "_MUSE.txt")

    return params, params2

def eval(trainer):
    src_emb = trainer.mapping(trainer.src_emb.weight).data
    tgt_emb = trainer.tgt_emb.weight.data

    out = []
    for method in ['nn', 'csls_knn_10']:
        results = get_word_translation_accuracy(
            trainer.src_dico.lang, trainer.src_dico.word2id, src_emb,
            trainer.tgt_dico.lang, trainer.tgt_dico.word2id, tgt_emb,
            method=method,
            dico_eval=trainer.params.dico_eval
        )
        out.append(results[0][1])
    return out


def joint_run(s2t_params, t2s_params, s2t_out=None, t2s_out=None):
    s2t_out = s2t_out if s2t_out is not None else s2t_params.out_file
    t2s_out = t2s_out if t2s_out is not None else t2s_params.out_file
    outfile = open(s2t_params.out_file, 'w')
    outfile.write("%s TO %s RUNS 1 TO %d\n" % (s2t_params.src_lang.upper(), s2t_params.tgt_lang.upper(), s2t_params.n_trials))
    outfile.close()
    outfile = open(t2s_params.out_file, 'w')
    outfile.write("%s TO %s RUNS 1 TO %d\n" % (t2s_params.src_lang.upper(), t2s_params.tgt_lang.upper(), t2s_params.n_trials))
    outfile.close()
    for i in range(params.n_trials):
        s2t_logger, s2t_trainer1, s2t_evaluator, s2t_outputs = run_model(s2t_params, i)
        t2s_logger, t2s_trainer1, t2s_evaluator, t2s_outputs = run_model(t2s_params, i)


        (proc_nn_s2t, proc_csls_s2t), (proc_nn_t2s, proc_csls_t2s), (joint_proc_nn_s2t, joint_proc_csls_s2t), \
        (joint_proc_nn_t2s, joint_proc_csls_t2s) = joint_procrustes(s2t_logger, s2t_trainer1, s2t_evaluator,
                                                                    t2s_logger, t2s_trainer1, t2s_evaluator,
                                                                    iters=s2t_params.n_refinement)

        s2t_outputs["proc_nn"] = proc_nn_s2t
        s2t_outputs["proc_csls"] = proc_csls_s2t
        s2t_outputs["joint_proc_nn"] = joint_proc_nn_s2t
        s2t_outputs["joint_proc_csls"] = joint_proc_csls_s2t
        t2s_outputs["proc_nn"] = proc_nn_t2s
        t2s_outputs["proc_csls"] = proc_csls_t2s
        t2s_outputs["joint_proc_nn"] = joint_proc_nn_t2s
        t2s_outputs["joint_proc_csls"] = joint_proc_csls_t2s

        save_model(s2t_params.out_file, s2t_outputs)
        save_model(t2s_params.out_file, t2s_outputs)


def run_model(params, runid):
    params.exp_name = params.src_lang + params.tgt_lang if params.exp_name is None else params.exp_name
    seed = np.random.randint(10000, 20000)
    params.seed = seed
    params.exp_id = str(runid)
    params.exp_path = ''
    # build model / trainer / evaluator
    logger = initialize_exp(params)
    src_emb, tgt_emb, mapping, discriminator = build_model(params, True)
    trainer = Trainer(src_emb, tgt_emb, mapping, discriminator, params)
    evaluator = Evaluator(trainer)

    base_nn, base_csls = _adversarial(params, logger, trainer, evaluator)

    outputs = {"run": runid, "seed": seed, "base_nn": base_nn, "base_csls": base_csls}

    return logger, trainer, evaluator, outputs

def save_model(file, scores):
    outfile = open(file, 'a')
    outfile.write("\t".join([k + ": " + str(v) for k, v in scores.items()]) + "\n")
    outfile.close()


def _adversarial(params, logger, trainer, evaluator):
    best_val = 0
    best_acc = 0
    for n_epoch in range(params.n_epochs):
        logger.info('Starting adversarial training epoch %i...' % n_epoch)
        tic = time.time()
        n_words_proc = 0
        stats = {'DIS_COSTS': []}

        for n_iter in range(0, params.epoch_size, params.batch_size):

            # discriminator training
            for _ in range(params.dis_steps):
                trainer.dis_step(stats)

            # mapping training (discriminator fooling)
            n_words_proc += trainer.mapping_step(stats)

            # log stats
            if n_iter % 6250 == 0:
                stats_str = [('DIS_COSTS', 'Discriminator loss')]
                stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                             for k, v in stats_str if len(stats[k]) > 0]
                stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
                logger.info(('%06i - ' % n_iter) + ' - '.join(stats_log))

                # reset
                tic = time.time()
                n_words_proc = 0
                for k, _ in stats_str:
                    del stats[k][:]

        # embeddings / discriminator evaluation
        to_log = OrderedDict({'n_epoch': n_epoch})
        evaluator.all_eval(to_log)
        evaluator.eval_dis(to_log)

        if to_log[VALIDATION_METRIC] > best_val:
            best_val = to_log[VALIDATION_METRIC]
            best_acc = eval(trainer)

        # JSON log / save best model / end of epoch
        logger.info("__log__:%s" % json.dumps(to_log))
        trainer.save_best(to_log, VALIDATION_METRIC)
        logger.info('End of epoch %i.\n\n' % n_epoch)

        # update the learning rate (stop if too small)
        trainer.update_lr(to_log, VALIDATION_METRIC)
        if trainer.map_optimizer.param_groups[0]['lr'] < params.min_lr:
            logger.info('Learning rate < 1e-6. BREAK.')
            break

    return best_acc

def procrustes(logger, trainer, evaluator, dico=None, iters=5):
    # Get the best mapping according to VALIDATION_METRIC
    logger.info('----> ITERATIVE PROCRUSTES REFINEMENT <----\n\n')
    trainer.reload_best()

    # training loop
    for n_iter in range(iters):
        logger.info('Starting refinement iteration %i...' % n_iter)

        # build a dictionary from aligned embeddings
        if n_iter == 0 and dico is not None:
            trainer.dico = dico
        else:
            trainer.build_dictionary()

        # apply the Procrustes solution
        trainer.procrustes()

        # embeddings evaluation
        #to_log = OrderedDict({'n_iter': n_iter})
        #evaluator.all_eval(to_log)

        # JSON log / save best model / end of epoch
        #logger.info("__log__:%s" % json.dumps(to_log))
        #trainer.save_best(to_log, VALIDATION_METRIC)
        logger.info('End of refinement iteration %i.\n\n' % n_iter)

    best_acc = eval(trainer)
    return best_acc

def joint_dicts(t1, t2):
    t1.build_dictionary()
    t2.build_dictionary()
    src_dico = t1.dico
    tgt_dico = t2.dico

    s2t_set = set([(a, b) for a, b in src_dico.cpu().numpy()])
    t2s_set = set([(b, a) for a, b in tgt_dico.cpu().numpy()])

    joint_set = s2t_set & t2s_set
    joint_dico = torch.LongTensor(list([[int(a), int(b)] for (a, b) in joint_set]))
    joint_dico = joint_dico.cuda()

    if len(joint_dico.size()) < 2:
        joint_dico = joint_dico.unsqueeze(1).expand(-1,2)
    joint_rev = joint_dico[:,[1,0]]

    return src_dico, tgt_dico, joint_dico, joint_rev

def joint_procrustes(l1, t1, e1, l2, t2, e2, iters=5):
    src, tgt, joint_src, joint_tgt = joint_dicts(t1, t2)

    src_scores = procrustes(l1, t1, e1, dico=src, iters=iters)
    tgt_scores = procrustes(l2, t2, e2, dico=tgt, iters=iters)
    src_joint_scores = procrustes(l1, t1, e1, dico=joint_src, iters=iters)
    tgt_joint_scores = procrustes(l2, t2, e2, dico=joint_tgt, iters=iters)

    return src_scores, tgt_scores, src_joint_scores, tgt_joint_scores


def save_output(file_name, accuracies):
    outfile = open(file_name, 'w')
    for i in range(len(accuracies)):
        outfile.write("\t".join([k+":"+str(v) for k,v in accuracies[i].items()]))
        outfile.write("\n")
    outfile.close()

if __name__ == '__main__':
    params = parse_args()

    s2t_params, t2s_params = set_default_args(params)
    joint_run(s2t_params, t2s_params, params.s2t_out, params.t2s_out)
