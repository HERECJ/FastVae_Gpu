from dataloader import RecData, UserItemData
from sampler_gpu_mm import SamplerBase, PopularSampler, MidxUniform, MidxUniPop
import torch
import torch.optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from vae_models import BaseVAE, VAE_Sampler
import argparse
import numpy as np
from utils import Eval
import utils
import logging
import datetime
import os
import time
import gc

def evaluate(model, train_mat, test_mat, logger, device):
    logger.info("Start evaluation")
    model.eval()
    with torch.no_grad():
        user_num, item_num = train_mat.shape
        
        user_emb = get_user_embs(train_mat, model, device)
        item_emb = model._get_item_emb()
        
        user_emb = user_emb.cpu().data
        item_emb = item_emb.cpu().data
        users = np.random.choice(user_num, min(user_num, 5000), False)
        m = Eval.evaluate_item(train_mat[users, :], test_mat[users, :], user_emb[users, :], item_emb, topk=50)
    return m

def get_user_embs(data_mat, model, device):
    data = UserItemData(data_mat, train_flag=False)
    dataloader = DataLoader(data, batch_size=config.batch_size_u, num_workers=config.num_workers, pin_memory=False, shuffle=False, collate_fn=utils.custom_collate_)
    user_lst = []
    for e in dataloader:
        user_his = e
        user_emb = model._get_user_emb(user_his.to(device))
        user_lst.append(user_emb)
    return torch.cat(user_lst, dim=0)

def train_model(model, train_mat, test_mat, config, logger):
    optimizer = utils_optim(config.learning_rate, model, config.weight_decay)
    scheduler = StepLR(optimizer, config.step_size, config.gamma)
    device = torch.device(config.device)


    train_data = UserItemData(train_mat)
    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True, shuffle=True, collate_fn=utils.custom_collate_)


    initial_list = []
    training_list = []
    sampling_list = []
    inference_list = []
    cal_loss_list = []
    for epoch in range(config.epoch):
        loss_ , kld_loss = 0.0, 0.0
        logger.info("Epoch %d"%epoch)
        if epoch > 0:
            del sampler
            if config.sampler > 2:
                del item_emb
        
        infer_total_time = 0.0
        sample_total_time = 0.0
        loss_total_time = 0.0
        t0 = time.time()
        if config.sampler > 0:
            if config.sampler == 1:
                sampler = SamplerBase(train_mat.shape[1] * config.multi, config.sample_num, device)
            elif config.sampler == 2:
                pop_count = np.squeeze(train_mat.sum(axis=0).A)
                pop_count = np.r_[pop_count, np.ones(train_mat.shape[1] * (config.multi -1))]
                sampler = PopularSampler(pop_count, config.sample_num, device)
            elif config.sampler == 3:
                item_emb = model._get_item_emb().detach()
                sampler = MidxUniform(item_emb, config.sample_num, device, config.cluster_num)
            elif config.sampler == 4:
                item_emb = model._get_item_emb().detach()
                pop_count = np.squeeze(train_mat.sum(axis=0).A)
                pop_count = np.r_[pop_count, np.ones(train_mat.shape[1] * (config.multi -1))]
                sampler = MidxUniPop(item_emb, config.sample_num, device, config.cluster_num, pop_count)
        t1 = time.time()
        
        
        for batch_idx, data in enumerate(train_dataloader):
            model.train()
            if config.sampler > 0 :
                sampler.train()
            else:
                sampler = None
            pos_id = data
            pos_id = pos_id.to(device)
            
            optimizer.zero_grad()
            tt0 = time.time()
            mu, logvar, loss, sample_time, loss_time = model(pos_id, sampler)
            tt1 = time.time()
            sample_total_time += sample_time
            infer_total_time += tt1 - tt0
            loss_total_time += loss_time
        
            kl_divergence = model.kl_loss(mu, logvar, config.anneal, reduction=config.reduction)/config.batch_size


            loss_ += loss.item()
            kld_loss += kl_divergence.item()
            loss += kl_divergence.item()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            # break
            # torch.cuda.empty_cache()
            t2= time.time()
        logger.info('--loss : %.2f, kl_dis : %.2f, total : %.2f '% (loss_, kld_loss, loss_ + kld_loss))
        torch.cuda.empty_cache()
        scheduler.step()
        gc.collect()
        initial_list.append(t1 - t0)
        training_list.append(t2 - t1)
        sampling_list.append(sample_total_time)
        inference_list.append(infer_total_time)
        cal_loss_list.append(loss_total_time)
        

        
        if (epoch % 10) == 0:
            result = evaluate(model, train_mat, test_mat, logger, device)
            logger.info('***************Eval_Res : NDCG@5,10,50 %.6f, %.6f, %.6f'%(result['item_ndcg'][4], result['item_ndcg'][9], result['item_ndcg'][49]))
            logger.info('***************Eval_Res : RECALL@5,10,50 %.6f, %.6f, %.6f'%(result['item_recall'][4], result['item_recall'][9], result['item_recall'][49]))
    logger.info('  Initial Time : {}'.format(np.mean(initial_list)))
    logger.info(' Sampling Time : {}'.format(np.mean(sampling_list)))
    logger.info(' Inference Time: {}'.format(np.mean(inference_list)))
    logger.info(' Calc Loss Time: {}'.format(np.mean(cal_loss_list)))
    logger.info(' Training Time (One epoch, including the dataIO, sampling, inference and backward time) : {}'.format(np.mean(training_list)))

def utils_optim(learning_rate, model, w):
    if config.optim=='adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=w)
    elif config.optim=='sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=w)
    else:
        raise ValueError('Unkown optimizer!')

def main(config, logger=None):
    device = torch.device(config.device)
    data = RecData(config.data_dir, config.data)
    train_mat, test_mat = data.get_data(config.ratio)
    user_num, item_num = train_mat.shape
    logging.info('The shape of datasets: %d, %d'%(user_num, item_num))

    assert config.sample_num < item_num
    
    if config.model == 'vae' and config.sampler == 0:
        model = BaseVAE(item_num * config.multi, config.dim)
    elif config.model == 'vae' and config.sampler > 0:
        model = VAE_Sampler(item_num * config.multi, config.dim)
    else:
        raise ValueError('Not supported model name!!!')
    model = model.to(device)
    train_model(model, train_mat, test_mat, config, logger)

    return evaluate(model, train_mat, test_mat, logger, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initialize Parameters!')
    parser.add_argument('-data', default='ml10M', type=str, help='path of datafile')
    parser.add_argument('-d', '--dim', default=[200, 32], type=int, nargs='+', help='the dimenson of the latent vector for student model')
    parser.add_argument('-s','--sample_num', default=500, type=int, help='the number of sampled items')
    parser.add_argument('--subspace_num', default=2, type=int, help='the number of splitted sub space')
    parser.add_argument('--cluster_num', default=16, type=int, help='the number of cluster centroids')
    parser.add_argument('-b', '--batch_size', default=256, type=int, help='the batch size for training')
    parser.add_argument('-e','--epoch', default=2, type=int, help='the number of epoches')
    parser.add_argument('-o','--optim', default='adam', type=str, help='the optimizer for training')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='the learning rate for training')
    parser.add_argument('--seed', default=20, type=int, help='random seed values')
    parser.add_argument('--ratio', default=0.8, type=float, help='the spilit ratio of dataset for train and test')
    parser.add_argument('--log_path', default='logs_test', type=str, help='the path for log files')
    parser.add_argument('--num_workers', default=16, type=int, help='the number of workers for dataloader')
    parser.add_argument('--data_dir', default='datasets', type=str, help='the dir of datafiles')
    parser.add_argument('--device', default='cuda', type=str, help='device for training, cuda or gpu')
    parser.add_argument('--model', default='vae', type=str, help='model name')
    parser.add_argument('--sampler', default=4, type=int, help='the sampler, 0 : no sampler, 1: uniform, 2: popular, 3: MidxUni, 4: MidxPop')
    parser.add_argument('--fix_seed', default=True, type=bool, help='whether to fix the seed values')
    parser.add_argument('--step_size', default=5, type=int, help='step size for learning rate discount')
    parser.add_argument('--gamma', default=0.95, type=float, help='discout for lr')
    parser.add_argument('--anneal', default=1.0, type=float, help='parameters for kl loss')
    parser.add_argument('--batch_size_u', default=128, type=int, help='batch size user for inference')
    parser.add_argument('--reduction', default=False, type=bool, help='loss if reduction')
    parser.add_argument('-w', '--weight_decay', default=1e-3, type=float, help='weight decay for the optimizer' )
    parser.add_argument('--multi', default=1, type=int, help='the number of extended items')


    config = parser.parse_args()

    import os
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    
    
    alg = config.model
    sampler = str(config.sampler) + '_' + str(config.multi) + 'x'
    ISOTIMEFORMAT = '%m%d-%H%M%S'
    timestamp = str(datetime.datetime.now().strftime(ISOTIMEFORMAT))
    loglogs = '_'.join((config.data, sampler, timestamp))
    log_file_name = os.path.join(config.log_path, loglogs)
    logger = utils.get_logger(log_file_name)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    logger.info(config)
    if config.fix_seed:
        utils.setup_seed(config.seed)
    import time
    t0 = time.time()
    m = main(config, logger)
    t1 = time.time()

    logger.info('Eval_Res : NDCG@5,10,50 %.6f, %.6f, %.6f'%(m['item_ndcg'][4], m['item_ndcg'][9], m['item_ndcg'][49]))
    logger.info('Eval_Res : RECALL@5,10,50 %.6f, %.6f, %.6f'%(m['item_recall'][4], m['item_recall'][9], m['item_recall'][49]))

    logger.info("Finish")
    svmat_name = log_file_name + '.mat'
    logger.info('Total Running Time: {}'.format(t1-t0) )