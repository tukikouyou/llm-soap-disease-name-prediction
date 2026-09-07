"""Convert the raw SOAP CSV into a plain-text corpus for language-model pretraining."""
import pandas as pd

csv_file_path = './03_SOAP(1).csv'
df = pd.read_csv(csv_file_path)

# Drop the ID column and flatten line breaks embedded in the note text.
df_without_first_column = df.iloc[:, 1:].applymap(lambda x: str(x).replace('\r', '').replace('\n', ''))

# Remove the header rows that the export repeats inside the data.
df_cleaned = df_without_first_column[~df_without_first_column.iloc[:, 0].str.contains("記載日")]

df_cleaned_no_spaces = df_cleaned.applymap(lambda x: str(x).replace(' ', ''))

txt_file_path_no_spaces = './output_no_spaces.txt'
df_cleaned_no_spaces.to_csv(txt_file_path_no_spaces, sep=' ', index=False, header=False)
