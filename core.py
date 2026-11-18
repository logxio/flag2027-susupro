"""Fixed feature standardization and trial-local bilinear forward pass."""
import numpy as np

def align(z, names):
    lookup = {str(n): j for j, n in enumerate(z['names'])}
    return np.column_stack([z['X'][:, lookup[n]] if n in lookup else
                            np.full(len(z['X']), np.nan) for n in names])

def fusion_design(head, z):
    x = align(z, head['names'])
    observed = np.isfinite(x)
    x = np.where(observed, (x-head['xmean'])/head['xsd'], 0.)
    x = np.clip(x, -8, 8)
    c = np.clip(np.nan_to_num((z['C']-head['cmean'])/head['csd']), -5, 5)
    gates = c[:, [0, 1, 2, 3, 4, 5, 6, 7]]
    interaction = (x[:, :, None]*gates[:, None, :]).reshape(len(x), -1)
    return np.column_stack([x, c, observed[:, head['availability']].astype(float), interaction])

def independent_predict(head, z):
    return np.einsum('nd,d->n', fusion_design(head, z), head['w'],
                     optimize=False)+head['bias']
