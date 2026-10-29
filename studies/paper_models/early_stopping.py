"""Patience on strict IoU improvements, with an auditable minimal loop edit."""
import ast
import inspect
import math


class EarlyStopping:
    def __init__(self, patience=10):
        if patience < 1:
            raise ValueError('patience must be positive')
        self.patience = patience
        self.best = -math.inf
        self.bad_epochs = 0
        self.observations = 0

    def update(self, value):
        if not math.isfinite(value):
            raise ValueError('nonfinite early-stopping monitor')
        self.observations += 1
        if value > self.best:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def install_train(common, patience=10):
    """Keep all original training statements; add monitor, break and stop record.

    Checkpoint >= tie behavior is retained. Only strict improvements reset the
    separate patience counter. The scheduler still uses args.epochs (now 100).
    """
    original = ast.parse(inspect.getsource(common.train)).body[0]
    tree = ast.parse(inspect.getsource(common.train))
    function = tree.body[0]
    loops = [(i,n) for i,n in enumerate(function.body)
             if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id == 'epoch']
    if len(loops) != 1:
        raise RuntimeError('Original training loop changed')
    index, loop = loops[0]
    initialize = ast.parse(f'early_stopper = EarlyStopping({patience})').body[0]
    stop = ast.parse('if early_stopper.update(best_iou):\n    break').body[0]
    record = ast.parse("write_json(records / 'early_stopping.json', dict(patience=early_stopper.patience, "
        "monitor='validation per-image foreground IoU', strict_improvement=True, "
        "completed_epochs=early_stopper.observations, maximum_epochs=args.epochs, "
        "bad_epochs=early_stopper.bad_epochs, stopped_early=early_stopper.bad_epochs >= early_stopper.patience, "
        "best_iou=early_stopper.best, scheduler_T_max=args.epochs))").body[0]
    loop.body.append(stop)
    function.body[index:index+1] = [initialize,loop,record]
    # Prove every original executable statement remains in its original order.
    import copy
    stripped = copy.deepcopy(function)
    del stripped.body[index+2]
    stripped.body[index+1].body.pop()
    del stripped.body[index]
    if ast.dump(stripped,include_attributes=False) != ast.dump(original,include_attributes=False):
        raise RuntimeError('Early stopping changed unrelated training statements')
    function.name = 'train_with_early_stopping'
    common.__dict__['EarlyStopping'] = EarlyStopping
    exec(compile(ast.fix_missing_locations(tree),'<baseline train + patience>', 'exec'),common.__dict__)
    return common.__dict__['train_with_early_stopping'],ast.unparse(tree)+'\n'
