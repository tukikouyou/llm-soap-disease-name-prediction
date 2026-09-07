"""Split the SOAP CSV into one TXT file per document, starting a new file at each key row."""
import csv

def split_csv_by_content(input_csv, key_column_index, key_value, output_prefix):
    with open(input_csv, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)

        header = next(reader)

        file_count = 1
        current_file = open(f'./results/{file_count}_{header[0]}.txt', 'w', encoding='utf-8')
        current_file.write(f'{header[1]}\t{header[2]}')

        for row in reader:
            # The key value marks the first row of a new document.
            if row[key_column_index] == key_value:
                current_file.close()
                file_count += 1
                current_file = open(f'./results/{file_count}_{row[0]}.txt', 'w', encoding='utf-8')

            current_file.write(f'{row[1]}\t{row[2]}')

        current_file.close()

split_csv_by_content('./03_SOAP(1).csv', 1, '記載日', 'output')
