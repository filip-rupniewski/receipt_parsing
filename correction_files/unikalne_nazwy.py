#read product_alias_dictionary.csv and print unique names from column correct_product_name
import csv

with open('product_alias_dictionary.csv', 'r') as file:
    reader = csv.DictReader(file, delimiter=';')
    unique_names_alias = set()
    for row in reader:
        unique_names_alias.add(row['correct_product_name'])

for idx, name in enumerate(sorted(unique_names_alias)):
    if name != "[IGNORE]":
        print(f"{idx};{name}")

print("\n========================================================================\n")

#now read product_translation_size.csv and print unique names from column original_name
with open('name_translation_size.csv', 'r') as file:
    reader = csv.DictReader(file, delimiter=';')
    unique_names_translation = set()
    for row in reader:
        unique_names_translation.add(row['original_name'])

for idx, name in enumerate(sorted(unique_names_translation)):
    if name != "[IGNORE]":
        print(f"{idx};{name}")

#now print unique_names_alias without unique_names_translation and other way around
print("\n unique_names_alias without unique_names_translation:\n")
for idx, name in enumerate(sorted(unique_names_alias - unique_names_translation)):
    print(f"{idx};{name}")
print("\nunique_names_translation without unique_names_alias:\n")
for idx, name in enumerate(sorted(unique_names_translation - unique_names_alias)):
    print(f"{idx};{name}")
