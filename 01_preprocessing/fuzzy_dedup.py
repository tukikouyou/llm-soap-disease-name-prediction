"""Similarity-based deduplication of a large text corpus, processed batch by batch."""
from difflib import SequenceMatcher
from tqdm import tqdm
from itertools import islice

def are_similar(a, b, threshold=0.8):
    """Return True when the two strings are more similar than the threshold."""
    return SequenceMatcher(None, a, b).ratio() > threshold

def remove_fuzzy_duplicates(lines, threshold=0.8):
    """Keep the first occurrence of every group of near-duplicate lines."""
    unique_lines = []

    for line in tqdm(lines, desc="Processing batch", unit="line"):
        line = line.strip()

        if not any(are_similar(line, unique_line, threshold) for unique_line in unique_lines):
            unique_lines.append(line)

    return unique_lines

def process_in_batches(input_file_path, output_file_path, batch_size=10000, threshold=0.8):
    """Deduplicate within each batch only; a global pass would be quadratic in corpus size."""
    all_batch_results = []
    with open(input_file_path, 'r', encoding='utf-8') as file:
        while True:
            lines = list(islice(file, batch_size))
            if not lines:
                break

            unique_batch_lines = remove_fuzzy_duplicates(lines, threshold)
            all_batch_results.extend(unique_batch_lines)

    save_to_file(output_file_path, all_batch_results)

def save_to_file(file_path, lines):
    with tqdm(total=len(lines), desc="Saving to file", unit="line") as pbar:
        with open(file_path, 'w', encoding='utf-8') as file:
            for line in lines:
                file.write(line + '\n')
                pbar.update(1)

input_file_path = 'output_file_1.txt'
output_file_path = 'output_file_2.txt'

process_in_batches(input_file_path, output_file_path, batch_size=3000, threshold=0.7)
