"""Apply Unicode normalization (NFKC by default) to a plain-text corpus file."""
import unicodedata

def normalize_text_file(file_path, output_path, normalization_form='NFC'):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        normalized_content = unicodedata.normalize(normalization_form, content)

        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(normalized_content)

    except Exception as e:
        print(f"Error: {e}")

input_file_path = r'./データ.txt'
output_file_path = r'./normalized_text.txt'
normalize_text_file(input_file_path, output_file_path, normalization_form='NFKC')
