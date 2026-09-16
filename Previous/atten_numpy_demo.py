import numpy as np

X = np.array([[[0.1, 0.2, 0.3, 0.4],
               [0.5, 0.4, 0.3, 0.2],
               [0.0, 0.1, 0.0, 0.1]]], dtype=np.float32)


# weight matrics 

Wq = np.array([[ 0.2, -0.1],
               [ 0.0,  0.1],
               [ 0.1,  0.2],
               [-0.1,  0.0]], dtype=np.float32)
Wk = np.array([[ 0.1,  0.1],
               [ 0.0, -0.1],
               [ 0.2,  0.0],
               [ 0.0,  0.2]], dtype=np.float32)
Wv = np.array([[ 0.1,  0.0],
               [-0.1,  0.1],
               [ 0.2, -0.1],
               [ 0.0,  0.2]], dtype=np.float32)

Q = X @ Wq
K = X @ Wk
V = X @ Wv

print("Q shape", Q.shape, "\nQ=n", Q[0])
print("K shape:", K.shape)
print("V shape", V.shape)

# Scaled dot-products 

scale = 1.0 / np.sqrt(Q.shape(-1))
attn_scores = (Q * K.transpose(0,2,1)) * scale


# casual mask (upper triangle set to -inf to softmax->0)

mask = np.triu(np.ones((1,3,3), dtype=bool), k = 1)
attn_scores = np.where(mask, -1e9, attn_scores)

#softmax over the last dim 

weights = np.exp(attn_scores - attn_scores.max(axis=-1, keepdims=True))

