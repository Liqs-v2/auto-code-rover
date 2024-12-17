import os
import re
from collections import deque
from functools import partial
import json

from app.data_structures import MessageThread
import torch
from datasets import load_dataset, Dataset
from pymilvus import MilvusClient
from transformers import AutoTokenizer, T5EncoderModel


def fetch_embedding_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    model = T5EncoderModel.from_pretrained("Salesforce/codet5-base")
    return model, tokenizer


def fetch_swe_bench_verified() -> Dataset:
    return load_dataset("princeton-nlp/SWE-bench_Verified", split='test')

def fetch_swe_bench_lite() -> Dataset:
    return load_dataset("princeton-nlp/SWE-bench_Lite", split='test')

def _count_tokens(batch):
    return {'tokens_in_problem_statement': [len(sample) for sample in batch['input_ids']]}


def split_text_by_paragraphs(text):
    """
    Split text into paragraphs based on 2 consecutive newlines in an OS-agnostic way. The resulting list
    of paragraphs does not include the newlines (ie captured groups).
    """
    return re.split(r'(?:\r?\n){2}|(?:\r){2}', text)


def chunk_paragraphs(paragraphs, tokenizer, chunk_size):
    """
    Chunk paragraphs into segments of approximately chunk_size tokens.
    Combines short paragraphs and splits long paragraphs intelligently.
    """
    chunks = []
    current_chunk = deque([], maxlen=chunk_size + 2)  # Use chunk_size+2, because we will need the special tokens too
    current_length = 0

    for paragraph in paragraphs:
        paragraph_tokens = tokenizer.encode(paragraph, add_special_tokens=False)
        paragraph_length = len(paragraph_tokens)

        if current_length + paragraph_length <= chunk_size:
            # We can fit this paragraph into the current chunk -> Add paragraph to current chunk
            current_chunk.extend(paragraph_tokens)
            current_length += paragraph_length
        else:
            # We cannot fit this paragraph into the current chunk, finalize the current chunk before moving on
            if current_chunk:
                current_chunk.insert(0, tokenizer.bos_token_id)
                current_chunk.append(tokenizer.eos_token_id)
                chunks.append(current_chunk)

            if paragraph_length <= chunk_size:
                # The paragraph fits into an entire chunk. Initialize the new chunk to this paragraph.
                current_chunk = deque(paragraph_tokens, maxlen=chunk_size + 2)
                current_length = len(paragraph_tokens)
            else:
                # The paragraph is too large for an entire chunk, process explicitly until the remaining tokens fit into
                # a single chunk
                for i in range(0, paragraph_length, chunk_size):
                    if paragraph_length - i > chunk_size:
                        # If we are still over the chunk size, keep generating full chunks
                        current_chunk = deque(paragraph_tokens[i:i + chunk_size], maxlen=chunk_size + 2)
                        current_chunk.insert(0, tokenizer.bos_token_id)
                        current_chunk.append(tokenizer.eos_token_id)
                        chunks.append(current_chunk)
                    else:
                        # If we are below the chunk size, we simply initialize the current chunk and its length
                        # with the remainder.
                        current_length = (paragraph_length - i) % chunk_size
                        current_chunk = deque(paragraph_tokens[i + chunk_size:], maxlen=chunk_size + 2)

    if current_chunk:
        current_chunk.insert(0, tokenizer.bos_token_id)
        current_chunk.append(tokenizer.eos_token_id)
        chunks.append(current_chunk)

    return chunks


def process_chunks(chunks, tokenizer, model):
    """
    Process each chunk through the model to obtain embeddings.
    """
    embeddings = []

    # Note that the chunks are dequeues of already tokenized paragraphs with bos and eos token ids pre and appended
    # This allows for very low level control when chunking. The drawback is that batching becomes more difficult,
    # since I cant use the tokenizer to generate paddings and attention_masks for me. However, since I am chunking
    # within problem_statement and want to build one embedding per problem_statement I need to keep track of
    # which problem_statement a chunk came from. The easiest way to do this is to simply process sequentially.
    # Considering my dataset is only 500 samples, this should be fine.
    for chunk in chunks:
        with torch.no_grad():
            outputs = model(torch.tensor([list(chunk)]))
            # Mean pooling for embeddings
            chunk_embedding = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(chunk_embedding)
    return embeddings


def aggregate_embeddings(embeddings):
    """
    Aggregate the embeddings obtained for each chunk.
    """
    return torch.mean(torch.stack(embeddings), dim=0)


def generate_embedding_for_sample(sample, model, tokenizer):
    paragraphs = split_text_by_paragraphs(sample['problem_statement'])
    chunks = chunk_paragraphs(paragraphs, tokenizer, chunk_size=model.config.n_positions - 2)
    embeddings = process_chunks(chunks, tokenizer, model)
    embedding = aggregate_embeddings(embeddings)
    return {
        'embedding': embedding.flatten()
    }


def main():
    # 1. Load SWE-Bench Verified from HuggingFace
    swe_bench_verified = fetch_swe_bench_verified()

    # 2. Extract `instance_id` and `problem_statement` from it
    swe_bench_verified = swe_bench_verified.remove_columns(
        [column for column in swe_bench_verified.column_names if column not in ['instance_id', 'problem_statement']]
    )

    # 3. Setup embedding model (CodeT5-base) from HF
    code_t5, code_t5_tokenizer = fetch_embedding_model_and_tokenizer()
    code_t5.eval()

    # Reading Note: Count tokens in problem statements to check context limit violations of embedder context limit
    # swe_bench_verified = swe_bench_verified.map(lambda batch: code_t5_tokenizer(batch['problem_statement']),
    #                                             batched=True, batch_size=16, add_special_tokens=False)
    #
    # swe_bench_verified = swe_bench_verified.map(_count_tokens, batched=True, batch_size=16)

    # 4. Generate embeddings for problem statements (use HF datasets instead of df here).
    generate_embedding_for_sample_fn = partial(
        generate_embedding_for_sample,
        tokenizer=code_t5_tokenizer,
        model=code_t5)

    swe_bench_verified = swe_bench_verified.map(generate_embedding_for_sample_fn, batched=False)
    swe_bench_verified = swe_bench_verified.rename_column('embedding', 'vector')
    swe_bench_verified = swe_bench_verified.add_column('id', [i for i in range(swe_bench_verified.num_rows)])
    swe_bench_verified = swe_bench_verified.add_column('trajectory', [''] * swe_bench_verified.num_rows)

    # # 5.    i. Extract old trajectories of ACR in LLM format
    # #       ii. Populate df with trajectories
    resolved_instance_ids = None
    excluded_repos = ['astropy', 'xarray', 'pylint']
    with open('results/acr-val-only/new_eval_results/report.json') as f:
        resolved_instance_ids = json.load(f)
        resolved_instance_ids = resolved_instance_ids['resolved']
        resolved_instance_ids = [repo for repo in resolved_instance_ids if not \
            any(excluded_repo in repo for excluded_repo in excluded_repos)]

    # Load SWE-Bench Lite and process only resolved trajectories
    swe_bench_lite = fetch_swe_bench_lite()
    swe_bench_lite = swe_bench_lite.remove_columns(
        [column for column in swe_bench_lite.column_names if column not in ['instance_id', 'problem_statement']]
    )
    swe_bench_lite = swe_bench_lite.filter(lambda example: example['instance_id'] in resolved_instance_ids)

    swe_bench_lite = swe_bench_lite.map(generate_embedding_for_sample_fn, batched=False)
    swe_bench_lite = swe_bench_lite.rename_column('embedding', 'vector')
    swe_bench_lite = swe_bench_lite.add_column('id', [i for i in range(len(resolved_instance_ids))])
    swe_bench_lite = swe_bench_lite.add_column('trajectory', [''] *  len(resolved_instance_ids))

    if not resolved_instance_ids:
        raise RuntimeError('Could not load report results, unable to initialize vector db with old ')

    for dirpath, dirnames, filenames in os.walk('results/acr-val-only/applicable_patch'):
        if  any(excluded_repo in dirpath for excluded_repo in excluded_repos) or not filenames:
            continue

        instance_id = re.match('^([^_]*__[^_]*)', dirpath.split('/')[-1]).group(0)

        if not instance_id in resolved_instance_ids:
            continue

        agent_write_patch_files = [filename for filename in filenames if 'debug_agent_write_patch' in filename]
        max_index = max([int(re.search(r'debug_agent_write_patch_(\d).json', file).group(1)) for file in agent_write_patch_files])
        msg_thread = MessageThread.load_from_file(os.path.join(dirpath, f'debug_agent_write_patch_{max_index}.json'))
        llm_trajectory = json.dumps(msg_thread.to_msg())

        swe_bench_lite = swe_bench_lite.map(lambda instance: {'trajectory': llm_trajectory} \
            if instance['instance_id'] == instance_id else instance)

    # # 6. Setup vector storage (just empty with right dims) and initialize with embeddings
    index_dimensions = code_t5.config.d_model

    client = MilvusClient('data/task_embeddings.db')
    client.create_collection(collection_name='swe_bench_verified', dimension=index_dimensions)
    client.insert(collection_name='swe_bench_verified', data=[dict(row) for row in swe_bench_verified])

    client.create_collection(collection_name='swe_bench_lite', dimension=index_dimensions)
    client.insert(collection_name='swe_bench_lite', data=[dict(row) for row in swe_bench_lite])

    # SQL style filtering
    # Reading Note: Seems like it only returns limit // 2 samples, so just double it to 1000 to consider all
    # client.load_collection('task_embeddings')

    # Reading Note: Read index in again
    client = MilvusClient('data/task_embeddings.db')
    client.load_collection('swe_bench_verified')
    client.load_collection('swe_bench_lite')

    client.get(collection_name='swe_bench_verified', ids=[0,499])
    client.get(collection_name='swe_bench_lite', ids=[0, 52])

    # client.query(collection_name='task_embeddings', limit=10, filter='instance_id LIKE "django%"')
    # Alternatively use get for direct access via index


if __name__ == "__main__":
    main()
