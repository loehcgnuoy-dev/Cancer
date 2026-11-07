import re, numpy as np, pandas as pd, scipy.sparse as sp
from itertools import chain
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

D = "data/"
train = pd.read_csv(D+"train.csv"); test = pd.read_csv(D+"test.csv")
sub = pd.read_csv(D+"sample_submission.csv")
cols = train.columns.drop(['ID','SUBCLASS']); G = len(cols)

def vtype(v):
    m = re.fullmatch(r'([A-Z])(\d+)([A-Z])', v)
    if m: return 1 if m[1]==m[3] else 2
    if '*' in v: return 3
    if 'fs' in v: return 4
    if 'splice' in v.lower(): return 5
    if re.search(r'del|ins|dup', v): return 6
    return 7

def explode(df):
    A = df[cols].fillna('WT').to_numpy(dtype=object)
    r, c = np.nonzero(A != 'WT')
    s = pd.Series(A[r, c]).str.split()
    n = s.str.len().to_numpy()
    return np.repeat(r, n), np.repeat(c, n), np.array(list(chain.from_iterable(s)), dtype=object)

def build(df, vocab=None):
    rr, cc, tok = explode(df)
    ut, inv = np.unique(tok, return_inverse=True)
    T = np.array([vtype(x) for x in ut])[inv]; n = len(df)
    def gm(mask):
        m = sp.coo_matrix((np.ones(mask.sum(), 'float32'), (rr[mask], cc[mask])), shape=(n, G)).tocsr()
        m.data[:] = 1; return m
    hits = sp.coo_matrix((np.ones((T>=2).sum(), 'float32'), (rr[T>=2], cc[T>=2])), shape=(n, G)).tocsr()
    nonsyn, trunc = gm(T>=2), gm(np.isin(T,[3,4,5]))
    agg = np.stack([np.bincount(rr[T==k], minlength=n) for k in range(1,8)], 1).astype('float32')
    nn = agg[:,1:].sum(1); tr = agg[:,2:5].sum(1); tot = agg.sum(1)
    agg = np.c_[agg, nn, tr, tr/(nn+1), agg[:,0]/(tot+1), agg[:,3]/(nn+1), agg[:,2]/(nn+1),
                np.log1p(tot), np.log1p(nn), tot==0, hits.max(1).toarray().ravel()>=2]
    key = pd.Series(np.array(cols)[cc]) + '_' + pd.Series(tok)
    if vocab is None:
        vc = key.value_counts(); vocab = vc[vc>=5].index.tolist()
    idx = {k:i for i,k in enumerate(vocab)}
    ki = key.map(idx); ok = ki.notna().to_numpy()
    H = sp.coo_matrix((np.ones(ok.sum(),'float32'), (rr[ok], ki[ok].astype(int))), shape=(n, len(vocab))).tocsr()
    return nonsyn, trunc, H, np.log1p(agg.astype('float32')), vocab

ns, tr_, H, agg, vocab = build(train)
ns_t, tr_t, H_t, agg_t, _ = build(test, vocab)

sc = StandardScaler().fit(agg)
A  = sp.csr_matrix(np.clip(sc.transform(agg), -5, 5))
A_t = sp.csr_matrix(np.clip(sc.transform(agg_t), -5, 5))
X  = sp.hstack([ns, tr_, H, A]).tocsr()
Xt = sp.hstack([ns_t, tr_t, H_t, A_t]).tocsr()

nz = np.diff(X.tocsc().indptr); keep = nz >= 5; keep[-A.shape[1]:] = True
X, Xt = X[:, keep], Xt[:, keep]

le = LabelEncoder(); y = le.fit_transform(train['SUBCLASS'])
model = LogisticRegression(C=0.3, max_iter=500, class_weight='balanced').fit(X, y)
pred = le.inverse_transform(model.predict(Xt))

sub['SUBCLASS'] = pd.Series(pred, index=test['ID']).reindex(sub['ID']).values
sub.to_csv('/mnt/user-data/outputs/submission_lr.csv', index=False, encoding='UTF-8-sig')
print(X.shape, Xt.shape, sub.shape, sub['SUBCLASS'].isna().sum())
print(sub['SUBCLASS'].value_counts().head(8).to_dict())
