import re
from collections import Counter


def apply_bpe(corpus, vocab_size):
    # Initialize an empty dictionary to store the BPE codes
    bpe_codes = {}

    # Split the corpus into individual words
    words = re.findall(r'\w+', corpus)

    # Initialize a frequency distribution to count the frequency of each word
    freq_dist = Counter(words)

    # Initialize a loop to repeat the following steps until the desired vocabulary size is reached
    while len(bpe_codes) < vocab_size:
        # Find the most frequent pair of consecutive symbols in the frequency distribution
        pair = max(freq_dist.items(), key=lambda x: x[1])

        # Add the merged pair to the BPE codes dictionary
        bpe_codes[pair[0]] = len(bpe_codes)

        # Replace all occurrences of the merged pair in the corpus with a new symbol that combines the merged pair
        new_symbol = ''.join(pair[0])
        words = [re.sub(''.join(pair[0]), new_symbol, word) for word in words]

        # Update the frequency distribution to count the frequency of the new symbol
        freq_dist = Counter(words)

    # Tokenize the corpus into individual BPE units using the BPE codes dictionary
    bpe_tokens = []
    for word in words:
        tokens = []
        i = 0
        while i < len(word):
            j = len(word)
            while j > i:
                token = word[i:j]
                if token in bpe_codes:
                    tokens.append(token)
                    i = j
                    break
                else:
                    j -= 1
            if j == i:
                tokens.append(word[i])
                i += 1
        bpe_tokens.append(tokens)

    return bpe_codes, bpe_tokens


print(apply_bpe("old " * 7 + "older " * 3 + "finest " * 9 + "lowest " * 4, 10))

#%%
