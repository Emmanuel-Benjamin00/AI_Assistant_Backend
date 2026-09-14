# About this AI Assistant

This AI Assistant is a retrieval-augmented generation (RAG) application. Instead of answering from the model's general knowledge, it answers questions using only the documents that have been indexed, and it shows the source passages it used.

Documents can be pasted as text or uploaded as PDF, DOCX, TXT or Markdown files. PDF text is extracted page by page, so every chunk from a PDF remembers its page number and answers can point to the page they came from. Scanned PDFs that contain only images have no extractable text and are rejected.

When a document is added, the backend splits it into chunks of up to about 1,200 characters. Paragraphs are kept together where possible, and very long paragraphs are split on sentence boundaries with a 200-character overlap so that no fact is lost at a boundary.

Each chunk is converted into a 1,536-dimension embedding vector with an OpenAI embedding model and stored in PostgreSQL using the pgvector extension. An HNSW index keeps similarity search fast as the number of chunks grows. PostgreSQL also stores a full-text search vector for every chunk, backed by a GIN index.

When a question is asked, the default hybrid search runs two searches: a vector search by cosine distance and a keyword search with PostgreSQL full-text search. The two ranked lists are merged with reciprocal rank fusion. An optional re-ranking step asks the chat model to score each candidate chunk for relevance and keeps the best ones.

The retrieved chunks are placed into a prompt that instructs the chat model to answer only from that context, to say it does not know when the answer is missing, and to cite chunk numbers. The answer is streamed to the browser token by token using Server-Sent Events, so the first words appear within about a second.

The assistant also has an agent mode that uses tool calling. The model can call tools to search the documents, list the indexed documents, or read a whole document, and it decides which tools to use for requests such as summarizing one document or comparing two documents.

The backend is built with Django and Django REST Framework, served by gunicorn on Azure App Service. The database is Azure Database for PostgreSQL. The frontend is a React application built with Vite and hosted on Azure Static Web Apps.

For production, the API rate-limits requests per client IP address, retries the AI provider on temporary failures, validates input sizes, locks document ingestion behind an access key on the public demo, and exposes a health check endpoint for the hosting platform.
