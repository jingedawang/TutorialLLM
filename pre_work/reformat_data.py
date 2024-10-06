import json
import random

# Load data.json
with open('data.txt', encoding='utf-8') as f:
    poems = json.load(f)

print(f'count={len(poems)}')

poems = [{
    'title': poem['title'],
    'author': poem['author'],
    'paragraphs': poem['paragraphs'],
} for poem in poems]

# Keep poems whose paragraphs have the same length
good_poems = []
for poem in poems:
    paragraphs = poem['paragraphs']
    if '□' in poem['title']:
        continue
    if '□' in ''.join(paragraphs):
        continue
    if len(set(len(paragraph) for paragraph in paragraphs)) > 1:
        continue
    if len(paragraphs) < 2:
        continue
    # Remove paragraphs with ()
    if any('(' in paragraph for paragraph in paragraphs) or any('（' in paragraph for paragraph in paragraphs) or '(' in poem['title'] or '（' in poem['title']:
        continue
    good_poems.append(poem)

print(f'count={len(good_poems)}')

# Save the data
json.dump(good_poems, open('data.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)