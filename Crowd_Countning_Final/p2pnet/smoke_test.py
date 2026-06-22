import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'p2pnet'))

from crowd_datasets.SHHA.loading_data import loading_data

ROOT = os.path.dirname(__file__)
train_list = os.path.join(ROOT, 'data', 'shanghaitech_p2p', 'combined_train.list')
val_list   = os.path.join(ROOT, 'data', 'shanghaitech_p2p', 'partA_test.list')

train_set, val_set = loading_data('.', train_lists=train_list, eval_list=val_list)
print(f'Train samples : {len(train_set)}')
print(f'Val   samples : {len(val_set)}')

img, target = train_set[0]
num_pts = sum(len(t['point']) for t in target)
print(f'img shape     : {img.shape}')
print(f'num points    : {num_pts}')
print('OK - 데이터 로더 정상')
