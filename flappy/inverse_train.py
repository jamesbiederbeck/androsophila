"""Trains the inverse flight-circuit model: given wing-muscle motor spike
counts, predict which haltere afferents were stimulated to cause them.

Plain single-hidden-layer MLP, manually implemented in numpy (no torch/etc
elsewhere in this repo, and ~(67*H + H*205) params is trivial without one).
input = wm motor spike counts (log1p + standardized), output = per-haltere-
afferent stimulation probability (independent sigmoids, trained with binary
cross-entropy) -- a "which cells to stimulate" proposal, not a claim about
exact currents. See flappy/inverse_data.py for how the training pairs were
generated and why this is a many-to-one, deliberately-not-unique inverse:
many different stimulated subsets can produce a similar motor response.
"""
import argparse
import glob
import json
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_dataset(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths: raise ValueError(f'No dataset files matched {pattern!r}')
    stims, responses, haltere_ids, wm_ids = [], [], None, None
    for p in paths:
        d = np.load(p)
        if haltere_ids is None:
            haltere_ids, wm_ids = d['haltere_ids'], d['wm_ids']
        elif not (np.array_equal(haltere_ids, d['haltere_ids']) and np.array_equal(wm_ids, d['wm_ids'])):
            raise ValueError(f'{p} uses different cell indexing than the first dataset file')
        stims.append(d['stim']); responses.append(d['response'])
    return np.concatenate(stims), np.concatenate(responses), haltere_ids, wm_ids


def features(response, mean, std):
    return (np.log1p(response.astype(np.float32)) - mean) / std


class InverseMLP:
    def __init__(self, n_in, n_hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, (2 / n_in) ** 0.5, (n_in, n_hidden)).astype(np.float32)
        self.b1 = np.zeros(n_hidden, dtype=np.float32)
        self.W2 = rng.normal(0, (2 / n_hidden) ** 0.5, (n_hidden, n_out)).astype(np.float32)
        self.b2 = np.zeros(n_out, dtype=np.float32)
        self._adam_state = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in self.params().items()}
        self._t = 0

    def params(self):
        return {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        h = np.maximum(z1, 0)
        logits = h @ self.W2 + self.b2
        return z1, h, logits

    def loss_and_grads(self, x, y):
        z1, h, logits = self.forward(x)
        p = 1 / (1 + np.exp(-logits))
        eps = 1e-7
        loss = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean()
        n = x.shape[0]
        dlogits = (p - y) / n
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dh = dlogits @ self.W2.T
        dz1 = dh * (z1 > 0)
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)
        return loss, p, {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def adam_step(self, grads, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        for k, g in grads.items():
            m, v = self._adam_state[k]
            m[:] = beta1 * m + (1 - beta1) * g
            v[:] = beta2 * v + (1 - beta2) * (g * g)
            mhat = m / (1 - beta1 ** self._t)
            vhat = v / (1 - beta2 ** self._t)
            getattr(self, k)[:] -= lr * mhat / (np.sqrt(vhat) + eps)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default=str(ROOT / 'outputs/flappy_inverse/dataset*.npz'))
    p.add_argument('--hidden', type=int, default=10000)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--val-frac', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=str(ROOT / 'outputs/flappy_inverse/inverse_model.npz'))
    args = p.parse_args()

    stim, response, haltere_ids, wm_ids = load_dataset(args.data)
    n = len(stim)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    n_val = max(1, int(n * args.val_frac))
    val_idx, train_idx = order[:n_val], order[n_val:]

    log_response = np.log1p(response[train_idx].astype(np.float32))
    mean, std = log_response.mean(axis=0), log_response.std(axis=0) + 1e-6

    x_train, y_train = features(response[train_idx], mean, std), stim[train_idx].astype(np.float32)
    x_val, y_val = features(response[val_idx], mean, std), stim[val_idx].astype(np.float32)

    model = InverseMLP(x_train.shape[1], args.hidden, y_train.shape[1], seed=args.seed)
    # Marginal-frequency baseline: predicting each cell's overall stimulation
    # rate regardless of input, to check the model learns more than that.
    base_p = np.clip(y_train.mean(axis=0), 1e-3, 1 - 1e-3)
    base_loss = -(y_val * np.log(base_p) + (1 - y_val) * np.log(1 - base_p)).mean()
    print(json.dumps({'n_train': len(train_idx), 'n_val': len(val_idx),
                       'n_in': x_train.shape[1], 'n_hidden': args.hidden, 'n_out': y_train.shape[1],
                       'marginal_baseline_val_bce': round(float(base_loss), 4)}))

    start = time.time()
    for epoch in range(args.epochs):
        perm = rng.permutation(len(x_train))
        epoch_loss = 0.
        for i in range(0, len(perm), args.batch_size):
            batch = perm[i:i + args.batch_size]
            loss, _, grads = model.loss_and_grads(x_train[batch], y_train[batch])
            model.adam_step(grads, args.lr)
            epoch_loss += loss * len(batch)
        epoch_loss /= len(x_train)
        val_loss, val_p, _ = model.loss_and_grads(x_val, y_val)
        val_acc = ((val_p > 0.5) == (y_val > 0.5)).mean()
        print(json.dumps({'epoch': epoch + 1, 'of': args.epochs, 'train_bce': round(float(epoch_loss), 4),
                           'val_bce': round(float(val_loss), 4), 'val_cellwise_accuracy': round(float(val_acc), 4),
                           'elapsed_s': round(time.time() - start, 1)}), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2,
             response_log_mean=mean, response_log_std=std,
             haltere_ids=haltere_ids, wm_ids=wm_ids,
             marginal_baseline_val_bce=base_loss)
    print(json.dumps({'status': 'done', 'out': str(out)}))


if __name__ == '__main__':
    main()
