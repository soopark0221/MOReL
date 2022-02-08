for ckpt_i, ckpt in enumerate(args.swag_ckpts):
    print("Checkpoint {}".format(ckpt))
    checkpoint = torch.load(ckpt)
    swag_model.subspace.rank = torch.tensor(0)
    swag_model.load_state_dict(checkpoint['state_dict'])

    for sample in range(args.swag_samples):
        swag_model.sample(.5)
        utils.bn_update(loaders['train'], swag_model)
        res = utils.predict(loaders['test'], swag_model)
        probs = res['predictions']
        targets = res['targets']
        nll = utils.nll(probs, targets)
        acc = utils.accuracy(probs, targets)

        if multiswag_probs is None:
            multiswag_probs = probs.copy()
        else:
            #TODO: rewrite in a numerically stable way
            multiswag_probs +=  (probs - multiswag_probs)/ (n_ensembled + 1)
        n_ensembled += 1

        ens_nll = utils.nll(multiswag_probs, targets)
        ens_acc = utils.accuracy(multiswag_probs, targets)
        values = [ckpt_i, sample, nll, acc, ens_nll, ens_acc]
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
        print(table)

print('Preparing directory %s' % args.savedir)
os.makedirs(args.savedir, exist_ok=True)
with open(os.path.join(args.savedir, 'eval_command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

np.savez(os.path.join(args.savedir, "multiswag_probs.npz"),
         predictions=multiswag_probs,
         targets=targets)