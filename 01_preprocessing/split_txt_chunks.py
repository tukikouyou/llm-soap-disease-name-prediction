"""Split a large corpus file into fixed-size chunks for downstream batch processing."""
import os

def split_txt_file(input_file_path, output_dir, lines_per_file=50000):
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    total_lines = len(lines)
    num_files = (total_lines // lines_per_file) + (1 if total_lines % lines_per_file != 0 else 0)

    for i in range(num_files):
        start = i * lines_per_file
        end = start + lines_per_file
        split_lines = lines[start:end]

        output_file_path = os.path.join(output_dir, f"split_file_{i+1}.txt")
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            outfile.writelines(split_lines)

input_file_path = './データ.txt'
output_dir = './soap_files'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

split_txt_file(input_file_path, output_dir)
