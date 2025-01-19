#include "ctorch.cnn.h"
#include "ctorch.h"
#include <cassert>

Cnn::CEmbedding::CEmbedding(int num_embeddings, int embedding_dim) : num_embeddings{num_embeddings}, embedding_dim{embedding_dim} {
    for (int i = 0; i < num_embeddings; i++) {
        embeddings.insert({i, CTorch::randn(embedding_dim)});
    }
}