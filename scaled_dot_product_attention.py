import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    # Step 1: Compute dot product of Q and K^T
    dot_product = np.dot(Q, K.T)

    # Step 2: Scale by sqrt of key dimension
    d_k = K.shape[-1]
    scaled_scores = dot_product / np.sqrt(d_k)

    # Step 3: Apply softmax to get attention weights
    attention_weights = softmax(scaled_scores)

    # Step 4: Multiply attention weights with V
    output = np.dot(attention_weights, V)

    return attention_weights, output

# Test Inputs
Q = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1]])
K = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1]])
V = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8]])

# Run attention
attention_weights, output = scaled_dot_product_attention(Q, K, V)

# Output results
print("1. Attention Weights Matrix (after softmax):")
print(attention_weights)
print("\n2. Final Output Matrix:")
print(output)
