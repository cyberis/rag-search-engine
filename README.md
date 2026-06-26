# Rag Search Engine
This is a cli based search feature that has been fortified through RAG processes. It is an MVP for RAG and search. I hope this doesn't take me forever.
Added Document Length correction.
TODO: Refactor commands to object methdods.
TODO: Finish implementing the optimized BM25 search.
Now with super duper BM25 search.
Now building semantic chunking and searching: DONE
Working on Hybrid Search
The LLM Code is atrocious. I mean the Instructor's solution. The embedding model returns Tensors but the instructor's solution expects np.array. That makes a mess of the whole code base. I wonder if it is because I am using GPU (RTX 4090) rather than running the inference on a CPU.