def build_dataset(args):
    if args.dataset_file == 'SHHA':
        from crowd_datasets.SHHA.loading_data import loading_data
        train_lists = getattr(args, 'train_list', None)
        eval_list   = getattr(args, 'val_list', None)
        return lambda data_root: loading_data(data_root, train_lists=train_lists, eval_list=eval_list)
    return None