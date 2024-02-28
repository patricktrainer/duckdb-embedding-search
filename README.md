## Overview
This repository contains a Python application that utilizes DuckDB as a backend to store and retrieve embedding vectors. The novel use of DuckDB allows for efficient similarity searches among large datasets. In this example, we've loaded comments from Hacker News and implemented functionality to find the 10 most similar comments to a given comment.

## Key Features
- **DuckDB Backend**: Utilizes DuckDB for efficient storage and retrieval of embedding vectors.
- **Embedding Vectors**: Embedding vectors are generated using OpenAI's models, ensuring high-quality semantic understanding.
- **Similarity Search**: Finds the most similar comments from a large dataset based on embedding comparisons.

## Getting Started

### Prerequisites
- Python 3.x
- DuckDB
- OpenAI API key

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/patricktrainer/duckdb-embedding-search.git
   ```

2. Navigate to the repository directory:
   ```sh
   cd duckdb-embedding-search
   ```
3. Install required packages:
   ```sh
   pip install -r requirements.txt
   ```

### Usage
To use the application, follow these steps:

1. **Set up your OpenAI API key**: Ensure you have your OpenAI API key set in your environment variables.
2. **Load the Comments**: Use `load_comments.py` to load comments into the DuckDB database. The comments and their corresponding embedding vectors will be stored in the `embeddings` table of the `hn_embeddings.db` database.
3. **Run the Similarity Search**: Execute the main script (e.g., `main.py`) and provide a Hacker News comment. The script will return the 10 most similar comments from the database.

> **Note** - The `get_similarity` function in `embedding.py` will create a new embedding vector for the provided comment if it does not already exist in the database. This means that it will hit the OpenAI API, which will count against your API usage.

#### Example results
The following example demonstrates the application's functionality. A comment is provided as input, and the application returns the 10 most similar comments from the database.

The comment provided as input:  
> One thing I’ve noticed is that many engineers, when they’re looking for a library on Github, they check the last commit time. They think that the more recent the last commit is, the better supported the library is. But what about an archived project that does exactly what you need it to do, has 0 bugs, and has been stable for years? That’s like finding a hidden gem in a thrift store! Most engineers I see nowadays will automatically discard a library that is not "constantly" updated... Implying it's a good thing :)


The most similar comments returned by the application (abbreviated for brevity):
1. > text: &gt; <i>Death to shared libraries. The headaches they cause are just not worth the benefit.</i><p>Completely disagree. Even though one size does not fit all, anyone who makes sweeping statements about static libraries is just stating to the world how they are completely oblivious regarding basic software maintenance problems such as tracking which software package is updated, specially those who are not kept up to date on a daily basis. 
    >
    > similarity: 0.8047998201033179

2. > text: Lots of good points here, but maintenance work for profitable systems seems like a valid use of time.<p>Now, some profitable systems are slowly bitrotting and tenured engineers can keep busy doing routine work while failing to address or escalate the bitrot. But I think people who are <i>good</i> at making sure boring and stable things stay boring and stable are usually underappreciated.
   > 
   > similarity: 0.796911347299464


## Architecture

### Modules
- `connection.py`: Handles DuckDB database connections.
- `embedding.py`: Manages embedding vector operations.
- `operations.py`: Contains utility functions for data processing.
- `openai_client.py`: Interfaces with the OpenAI API.

### DuckDB Integration
DuckDB is used as a lightweight, high-performance database to store embedding vectors. The `connection.py` module establishes a connection to DuckDB, and `operations.py` contains the logic for inserting and retrieving embeddings.

### Embedding Vectors
Embedding vectors are generated using OpenAI's API. The `openai_client.py` module contains the logic for interfacing with the API. The `embedding.py` module contains the logic for generating embedding vectors and comparing them.
