import torch.optim as optim


config = {}
config['data_dir'] = 'datasets/'
config['dataset'] = '1000'
config['train_img_dir'] = 'VQAimages/train2014'
config['val_img_dir'] = 'VQAimages/val2014'
config['test_img_dir'] = 'VQAimages/test2015'
config['em_fname'] = 'embeddings/glove.6B.300d.txt'
config['train'] = True
config['model'] = 'JointEmbedModel'
config['maxlen'] = 30
config['batch_size'] = 2
config['num_workers'] = 0
config['pin_memory'] = False

jointembedmodel_cfg = {}
jointembedmodel_cfg['lr'] = 0.0001
jointembedmodel_cfg['L2'] = 0.05
jointembedmodel_cfg['opt'] = optim.Adam
jointembedmodel_cfg['epochs'] = 10
jointembedmodel_cfg['early_stop'] = True
jointembedmodel_cfg['early_stop_epoch'] = 3
jointembedmodel_cfg['lr_decay'] = 0.98
jointembedmodel_cfg['GPU'] = False
jointembedmodel_cfg['GPU_Ids'] = [0, 1]
jointembedmodel_cfg['embedding_dim'] = 300
