# RAG Techniques Glossary

Retrieval-augmented generation (RAG) combines a search step with a language model. The search step finds passages related to the question, and the model writes an answer from those passages. RAG lets a model answer from private or recent documents without fine-tuning the model.

An embedding is a list of numbers that represents the meaning of a piece of text. Texts with similar meanings have embeddings that point in similar directions, so cosine similarity between two embeddings measures how related the texts are. Vector search is good at matching paraphrases, such as a question about "money back" finding a passage about refunds.

Keyword search matches the actual words in the query. PostgreSQL full-text search turns text into a tsvector of normalized word stems and ranks matches with functions such as ts_rank_cd. BM25 is the classic keyword ranking formula used by search engines like Elasticsearch. Keyword search is good at exact terms that embeddings can blur, such as codes, product names and IDs.

Hybrid search runs vector search and keyword search together and merges the results. Reciprocal rank fusion (RRF) is a simple way to merge ranked lists: each result scores 1 divided by (k + rank) in every list it appears in, the scores are added, and k is usually 60. RRF only uses ranks, so it does not need the two searches to have comparable scores.

Re-ranking is a second, more careful pass over the top candidates from the first search. A cross-encoder model or an LLM reads the question and each candidate together and scores how well the candidate answers the question. Re-ranking improves precision but adds latency and cost, so it is applied only to a small candidate set.

Chunking splits documents into passages before embedding. Chunks that are too large mix several topics and dilute the embedding, while chunks that are too small lose context. Overlap between neighbouring chunks keeps a fact intact when it falls on a boundary.

RAG systems are evaluated at two levels. Retrieval metrics include hit rate, the share of questions where a relevant chunk appears in the top k results, and mean reciprocal rank (MRR), the average of 1 divided by the rank of the first relevant chunk. Answer metrics include faithfulness, whether every claim in the answer is supported by the retrieved context, and correctness against a reference answer. An LLM can act as a judge to score faithfulness at scale.

Tool calling lets a model request a function call instead of replying with text. The application runs the function, sends the result back, and the model continues until it can answer. An agent is a loop of model calls and tool calls, and it needs a step limit so it cannot loop forever.
