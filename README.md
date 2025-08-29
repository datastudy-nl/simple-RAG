# simple-RAG

A tiny, no-magic Retrieval-Augmented Generation (RAG) pipeline in pure Python. It uses Sentence-Transformers for embeddings, FAISS for retrieval, and talks to an Ollama LLM for generation. Clean, minimal, and easy to extend—perfect for learning or as a foundation for your own projects.

## Highlights

- Pure Python, no LangChain
- Sentence-Transformers embeddings + FAISS vector search
- Simple text/PDF ingestion with chunking and overlap
- Streams responses from a local Ollama server
- Small, readable codebase designed for teaching and hacking

## Project structure

```
simple-RAG/
├─ main.py              # CLI entrypoint; wires retrieval to Ollama
├─ rag.py               # Chunking, embeddings, FAISS store & retrieval
├─ fileutils.py         # File loading utilities (PDF/TXT/MD)
├─ knowledge_base/      # Your source documents live here
│  ├─ shrek.txt
│  └─ bee_movie_script.txt
└─ requirements.txt     # Python dependencies
```

## How it works

1. Load documents from `knowledge_base/` (PDF, TXT, MD).
2. Split into overlapping chunks for better context.
3. Embed chunks with Sentence-Transformers.
4. Build a FAISS index for fast similarity search.
5. At query time, retrieve top-k chunks and pass them to the LLM via Ollama.

## Requirements

- Python 3.10+
- A working [Ollama](https://ollama.com) installation running locally (default: `http://localhost:11434`).
- An Ollama model downloaded (e.g., `mistral`).

## Installation

Create and activate a virtual environment, then install dependencies.

```powershell
python -m venv env
./env/Scripts/Activate.ps1
pip install -r requirements.txt
```

Pull a model for Ollama (example: mistral):

```powershell
ollama pull mistral
```

## Quickstart

1. Put your documents into `knowledge_base/` as `.txt`, `.md`, or `.pdf` files.
2. Start Ollama (if it isn’t already running).
3. Run the app:

```powershell
python main.py
```

Ask questions interactively. Type `exit` to quit.

## Configuration

Key constants you may want to tweak:

- In `rag.py`:
	- `CHUNK_SIZE` (default: 1000)
	- `CHUNK_OVERLAP` (default: 100)
	- `MODEL_NAME` (default: `sentence-transformers/all-MiniLM-L6-v2`)
	- `FAISS_INDEX_PATH` / `DOCS_PATH`
- In `main.py`:
	- `OLLAMA_URL` (default: `http://localhost:11434/api/generate`)
	- `OLLAMA_MODEL` (default: `mistral`)

## Commands and usage

- Start interactive RAG session:
	```powershell
	python main.py
	```

## Development

- Style: keep functions small and well documented.
- Tests: this repo is tiny; consider adding smoke tests as you extend it.
- Contributions: PRs and issues are welcome.

### Extending

- Add persistence: call `VectorStore.save()`/`load()` to reuse the index.
- Swap embedding model: change `MODEL_NAME` in `rag.py`.
- Change retriever behavior: adjust `k` in `store.search(query, k=3)`.
- Add sources formatting: currently prints the first 200 chars of each chunk.

## Troubleshooting

- Import errors for packages (requests, pypdf, sentence-transformers, faiss-cpu, numpy):
	- Ensure your virtual environment is active and run `pip install -r requirements.txt`.
- Ollama connection errors:
	- Verify the service is running and reachable at `OLLAMA_URL`.
	- Confirm the model is available (`ollama list`) and pulled (`ollama pull mistral`).
- GPU vs CPU FAISS:
	- This project pins CPU FAISS via `faiss-cpu`. If you have a GPU and want acceleration, install a suitable FAISS build manually.

## FAQ

Q: Can I use another LLM provider?

A: Yes. Replace `query_ollama()` in `main.py` with a function that calls your provider, keeping the same input/output signature.

Q: How big can my documents be?

A: As big as your memory allows. The index holds embeddings for each chunk; large corpora will use more RAM and take time to build. Consider batching or on-disk indices for very large datasets.

Q: Why chunk overlap?

A: Overlap helps preserve context that might otherwise be split between chunks, improving retrieval quality.

## Security & privacy

By default, all data stays local: files, embeddings, and LLM calls (with Ollama). Review your model’s behavior and logs before sharing outputs.

## License

MIT License. See `LICENSE` for details.
