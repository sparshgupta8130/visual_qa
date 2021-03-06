import torch.optim as optim


config = {}
config['data_dir'] = 'datasets/'
config['dataset'] = '3000'
config['train_img_dir'] = 'VQAimages/train2014'
config['val_img_dir'] = 'VQAimages/val2014'
config['test_img_dir'] = 'VQAimages/test2015'
config['em_fname'] = 'embeddings/glove.6B.300d.txt'
config['train'] = True
config['model'] = 'MultihopAttentionModel'
config['test_model'] = 'saved_models/2019-03-18__10h22m20s_MultihopAttentionModel3000_.pt'
config['maxlen'] = 30
config['batch_size'] = 1024
config['num_workers'] = 8
config['pin_memory'] = True

jointembedmodel_cfg = {}
jointembedmodel_cfg['lr'] = 0.0005
jointembedmodel_cfg['L2'] = 0
jointembedmodel_cfg['opt'] = optim.Adam
jointembedmodel_cfg['epochs'] = 15
jointembedmodel_cfg['early_stop'] = True
jointembedmodel_cfg['early_stop_epoch'] = 3
jointembedmodel_cfg['lr_decay'] = 0.98
jointembedmodel_cfg['GPU'] = True
jointembedmodel_cfg['GPU_Ids'] = [0, 1, 2, 3]
jointembedmodel_cfg['embedding_dim'] = 300

attentionmodel_cfg = {}
attentionmodel_cfg['lr'] = 0.001
attentionmodel_cfg['L2'] = 0
attentionmodel_cfg['opt'] = optim.Adam
attentionmodel_cfg['epochs'] = 15
attentionmodel_cfg['early_stop'] = True
attentionmodel_cfg['early_stop_epoch'] = 3
attentionmodel_cfg['lr_decay'] = 0.98
attentionmodel_cfg['GPU'] = True
attentionmodel_cfg['GPU_Ids'] = [0, 1, 2, 3]
attentionmodel_cfg['embedding_dim'] = 300
attentionmodel_cfg['dropout'] = 0.5
attentionmodel_cfg['glimpses'] = 2

jointembedresnet_cfg = {}
jointembedresnet_cfg['lr'] = 0.0005
jointembedresnet_cfg['L2'] = 0
jointembedresnet_cfg['opt'] = optim.Adam
jointembedresnet_cfg['epochs'] = 15
jointembedresnet_cfg['early_stop'] = True
jointembedresnet_cfg['early_stop_epoch'] = 3
jointembedresnet_cfg['lr_decay'] = 0.98
jointembedresnet_cfg['GPU'] = True
jointembedresnet_cfg['GPU_Ids'] = [0, 1, 2, 3]
jointembedresnet_cfg['embedding_dim'] = 300

multihopmodel_cfg = {}
multihopmodel_cfg['lr'] = 0.0001
multihopmodel_cfg['L2'] = 0
multihopmodel_cfg['opt'] = optim.Adam
multihopmodel_cfg['epochs'] = 15
multihopmodel_cfg['early_stop'] = True
multihopmodel_cfg['early_stop_epoch'] = 3
multihopmodel_cfg['lr_decay'] = 0.98
multihopmodel_cfg['GPU'] = True
multihopmodel_cfg['GPU_Ids'] = [0, 1, 2, 3]
multihopmodel_cfg['embedding_dim'] = 300
multihopmodel_cfg['dropout'] = 0.5
multihopmodel_cfg['hops'] = 2