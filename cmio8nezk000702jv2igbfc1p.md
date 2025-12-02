---
title: "Building LLM's From Scratch"
datePublished: Tue Jul 01 2025 05:38:24 GMT+0000 (Coordinated Universal Time)
cuid: cmio8nezk000702jv2igbfc1p
slug: building-llms-from-scratch--deleted
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1764659407401/bf63eee6-9c68-4a60-9cc7-0bf4e126c79f.webp

---

**Core Components:**
1. **Data Download & Preprocessing** - Downloads text from Project Gutenberg with fallback
2. **SimpleTokenizer** - Word-based tokenization with vocabulary building
3. **TextDataset** - PyTorch dataset for sequence-to-sequence training
4. **PositionalEncoding** - Adds position information to embeddings
5. **MultiHeadAttention** - Core attention mechanism
6. **FeedForward** - Feed-forward neural network
7. **TransformerBlock** - Complete transformer layer
8. **SimpleGPT** - Full GPT model
9. **GPTTrainer** - Training loop with validation
10. **Text Generation** - Advanced text generation with top-k sampling

**Key Features:**
- Automatic device detection (MPS/CUDA/CPU)
- Proper weight initialization
- Gradient clipping and learning rate scheduling
- Model checkpointing
- Causal masking for autoregressive generation

**Usage:**
Simply run `python filename.py` and it will:
1. Download/prepare the dataset
2. Build the tokenizer
3. Create and train the model
4. Save the complete model
5. Generate sample text

---

## Step 1: Initial Setup and Device Detection

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
# ... other imports

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

**What's happening here:**

1. **Random Seeds**: Setting seeds to 42 ensures reproducible results
   - Every time you run the code, you'll get the same "random" numbers
   - This makes debugging easier and results consistent

2. **Device Detection**: 
   - **MPS (Metal Performance Shaders)**: Apple's GPU acceleration for M1/M2 Macs
   - **CUDA**: NVIDIA GPU acceleration
   - **CPU**: Fallback for any computer

**Why this matters:**
- GPUs are much faster for matrix operations (10-100x speedup)
- Your model will automatically use the best available hardware

**What gets stored:**
```python
device = torch.device("mps")  # or "cuda" or "cpu"
```

**Key Concept - Tensors:**
- All data in PyTorch is stored as "tensors" (multi-dimensional arrays)
- Tensors can live on CPU or GPU
- Example:
```python
# CPU tensor
x = torch.tensor([1, 2, 3])

# GPU tensor (much faster for large operations)
x_gpu = torch.tensor([1, 2, 3]).to(device)
```

**Questions to check understanding:**
1. Why do we set random seeds?
2. What's the difference between CPU and GPU processing?
3. What is a tensor?

---

## Step 2: Data Download and Preprocessing

```python
def download_dataset():
    Path("data").mkdir(exist_ok=True)
    url = "https://www.gutenberg.org/files/11/11-0.txt"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text = response.text
        
        start_idx = text.find("*** START OF")
        end_idx = text.find("*** END OF")
        
        if start_idx != -1 and end_idx != -1:
            clean_text = text[start_idx:end_idx]
            newline_idx = clean_text.find('\n\n')
            if newline_idx != -1:
                clean_text = clean_text[newline_idx + 2:]
        else:
            clean_text = text
        
        clean_text = clean_text[:50000]  # Take first 50,000 characters
        
        with open('data/dataset.txt', 'w', encoding='utf-8') as f:
            f.write(clean_text)
        
        return clean_text
```

**What's happening step by step:**

### 1. **Create Data Directory**
```python
Path("data").mkdir(exist_ok=True)
```
- Creates a folder called "data" if it doesn't exist
- `exist_ok=True` means "don't crash if folder already exists"

### 2. **Download Raw Text**
```python
url = "https://www.gutenberg.org/files/11/11-0.txt"
response = requests.get(url, timeout=30)
```
- Downloads "Alice's Adventures in Wonderland" from Project Gutenberg
- Project Gutenberg = free digital library of books
- File 11 = Alice in Wonderland (classic text for ML experiments)

### 3. **Clean the Text**
**Raw downloaded text looks like:**
```
The Project Gutenberg eBook of Alice's Adventures in Wonderland

*** START OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***

Alice was beginning to get very tired of sitting by her sister on the bank...

[ACTUAL STORY CONTENT]

*** END OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***

End of the Project Gutenberg EBook...
```

**Cleaning process:**
```python
start_idx = text.find("*** START OF")  # Find where story begins
end_idx = text.find("*** END OF")      # Find where story ends
clean_text = text[start_idx:end_idx]   # Extract only the story part
```

### 4. **Final Processing**
```python
clean_text = clean_text[:50000]  # Take first 50,000 characters
```
- Limits size to 50K characters for faster training on laptops
- Full Alice in Wonderland is ~150K characters

**What gets saved to disk:**
```
data/dataset.txt
├── Content: "Alice was beginning to get very tired of sitting by her sister..."
├── Size: ~50,000 characters
└── Format: Plain text, UTF-8 encoding
```

**Example of cleaned text:**
```
"Alice was beginning to get very tired of sitting by her sister on the bank, 
and of having nothing to do: once or twice she had peeped into the book her 
sister was reading, but it had no pictures or conversations in it, 'and what 
is the use of a book,' thought Alice 'without pictures or conversation?'"
```

### 5. **Fallback Data (if download fails)**
```python
fallback_text = """
The quick brown fox jumps over the lazy dog...
Alice was beginning to get very tired...
""" * 100
```
- If internet fails, uses simple repeated sentences
- Ensures code always works, even offline

**Key Concepts:**
1. **Text Preprocessing**: Cleaning raw data before feeding to ML models
2. **Character vs Word Count**: 50K characters ≈ 8-10K words
3. **UTF-8 Encoding**: Standard way to store text with special characters

**What you now have:**
- A clean text file with story content
- No headers, footers, or metadata
- Ready for the next step: tokenization

**Memory representation:**
```python
clean_text = "Alice was beginning to get very tired..."
# Type: string
# Length: 50,000 characters
# Storage: ~50KB in memory
```
---

![Image description](https://cdn.hashnode.com/res/hashnode/image/upload/v1764659405014/25bb8cd6-8a17-41d9-8766-bfdd8e375b10.png)

## Step 3: Tokenization - Converting Text to Numbers

This is **CRUCIAL** - neural networks can't understand text, only numbers! We need to convert words to numbers.

```python
class SimpleTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.word_to_int = {}    # Dictionary: word -> number
        self.int_to_word = {}    # Dictionary: number -> word  
        self.vocab = []          # List of all words
        self.word_freq = Counter()  # How often each word appears
```

### **Step 3A: Text Cleaning**

```python
def clean_text(self, text):
    text = text.lower()  # "Alice" -> "alice"
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\:\-\'\"]', ' ', text)
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces -> single space
    text = text.strip()
    return text
```

**What happens:**
```
Input:  "Alice was VERY tired!!! She thought, 'This is boring...'"
Step 1: "alice was very tired!!! she thought, 'this is boring...'"
Step 2: "alice was very tired    she thought   this is boring   "
Step 3: "alice was very tired she thought this is boring"
Output: "alice was very tired she thought this is boring"
```

### **Step 3B: Build Vocabulary**

```python
def build_vocab(self, text):
    clean_text = self.clean_text(text)
    words = clean_text.split()  # Split into individual words
    self.word_freq = Counter(words)  # Count frequency of each word
```

**Example word counting:**
```python
text = "alice was tired alice was very tired"
words = ["alice", "was", "tired", "alice", "was", "very", "tired"]

word_freq = Counter(words)
# Result: {'alice': 2, 'was': 2, 'tired': 2, 'very': 1}
```

**Create final vocabulary:**
```python
special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
most_common_words = self.word_freq.most_common(vocab_size - 4)

self.vocab = special_tokens + [word for word, _ in most_common_words]
```

**Special tokens explained:**
- `<PAD>`: Padding (fill empty spaces)
- `<UNK>`: Unknown word (not in vocabulary)
- `<BOS>`: Beginning of sequence
- `<EOS>`: End of sequence

**Example vocabulary (first 10 items):**
```python
vocab = [
    '<PAD>',    # ID: 0
    '<UNK>',    # ID: 1  
    '<BOS>',    # ID: 2
    '<EOS>',    # ID: 3
    'the',      # ID: 4 (most common word)
    'and',      # ID: 5
    'to',       # ID: 6
    'a',        # ID: 7
    'alice',    # ID: 8
    'was',      # ID: 9
    # ... up to vocab_size=800 words
]
```

### **Step 3C: Create Word-to-Number Mappings**

```python
self.word_to_int = {word: i for i, word in enumerate(self.vocab)}
self.int_to_word = {i: word for i, word in enumerate(self.vocab)}
```

**Resulting dictionaries:**
```python
word_to_int = {
    '<PAD>': 0,
    '<UNK>': 1,
    '<BOS>': 2,
    '<EOS>': 3,
    'the': 4,
    'and': 5,
    'alice': 8,
    'was': 9,
    # ... 800 total words
}

int_to_word = {
    0: '<PAD>',
    1: '<UNK>',
    2: '<BOS>', 
    3: '<EOS>',
    4: 'the',
    5: 'and',
    8: 'alice',
    9: 'was',
    # ... 800 total words
}
```

### **Step 3D: Encoding (Text → Numbers)**

```python
def encode(self, text):
    clean_text = self.clean_text(text)
    words = clean_text.split()
    
    numbers = []
    for word in words:
        if word in self.word_to_int:
            numbers.append(self.word_to_int[word])
        else:
            numbers.append(self.word_to_int['<UNK>'])  # Unknown word
    
    return numbers
```

**Example encoding:**
```python
text = "Alice was tired"
clean_text = "alice was tired"
words = ["alice", "was", "tired"]

# Look up each word:
numbers = [
    word_to_int["alice"],  # 8
    word_to_int["was"],    # 9  
    word_to_int["tired"]   # 45 (assuming "tired" is 45th most common)
]

result = [8, 9, 45]
```

### **Step 3E: Decoding (Numbers → Text)**

```python
def decode(self, numbers):
    words = []
    for num in numbers:
        if num in self.int_to_word:
            words.append(self.int_to_word[num])
        else:
            words.append('<UNK>')
    
    return ' '.join(words)
```

**Example decoding:**
```python
numbers = [8, 9, 45]

# Look up each number:
words = [
    int_to_word[8],   # "alice"
    int_to_word[9],   # "was"  
    int_to_word[45]   # "tired"
]

result = "alice was tired"
```

### **What Gets Saved:**

```python
# File: data/tokenizer.pkl
tokenizer_data = {
    'vocab_size': 800,
    'word_to_int': {'<PAD>': 0, '<UNK>': 1, ..., 'tired': 45, ...},
    'int_to_word': {0: '<PAD>', 1: '<UNK>', ..., 45: 'tired', ...},
    'vocab': ['<PAD>', '<UNK>', '<BOS>', '<EOS>', 'the', 'and', ...],
    'word_freq': Counter({'the': 1234, 'and': 987, 'alice': 156, ...})
}
```

### **Memory Format:**

**Before tokenization:**
```
"Alice was beginning to get very tired" (string, ~37 characters)
```

**After tokenization:**
```
[8, 9, 234, 4, 67, 12, 45] (list of integers, 7 numbers)
```

### **Why This Matters:**
1. **Neural networks only understand numbers**
2. **Consistent mapping**: Same word always gets same number
3. **Vocabulary size controls model complexity**: 800 words = manageable for small model
4. **Unknown words handled gracefully**: Rare words become `<UNK>`

### **Key Insight:**
Your entire book is now represented as a sequence of numbers between 0 and 799!

```
Original: "Alice was beginning to get very tired of sitting by her sister..."
Tokenized: [8, 9, 234, 4, 67, 12, 45, 23, 156, 34, 89, 234, ...]
```
---
## Step 4: Creating the Training Dataset

Now we need to convert our tokenized text into **training examples** that teach the model to predict the next word.

```python
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_length=32):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        self.tokens = tokenizer.encode(text)  # Convert entire text to numbers
        self.examples = []
        
        for i in range(len(self.tokens) - seq_length):
            input_seq = self.tokens[i:i + seq_length]
            target_seq = self.tokens[i + 1:i + seq_length + 1]
            self.examples.append((input_seq, target_seq))
```

### **Step 4A: Understanding the Core Concept**

**The key insight:** To predict the next word, the model learns from **input → target** pairs where target is input shifted by 1 position.

**Example with small sequence:**
```python
# Original tokenized text:
tokens = [8, 9, 234, 4, 67, 12, 45, 23, 156, 34, 89, 67, 234, 445, 23]
#        alice was beginning to get very tired of sitting by her get beginning long of

# With seq_length = 5, we create these training examples:
```

### **Step 4B: Creating Training Examples (Sliding Window)**

```python
seq_length = 5  # Model sees 5 words at once

# Example 1:
i = 0
input_seq = tokens[0:5]    # [8, 9, 234, 4, 67]     "alice was beginning to get"
target_seq = tokens[1:6]   # [9, 234, 4, 67, 12]   "was beginning to get very"

# Example 2:
i = 1  
input_seq = tokens[1:6]    # [9, 234, 4, 67, 12]   "was beginning to get very"
target_seq = tokens[2:7]   # [234, 4, 67, 12, 45]  "beginning to get very tired"

# Example 3:
i = 2
input_seq = tokens[2:7]    # [234, 4, 67, 12, 45]  "beginning to get very tired"
target_seq = tokens[3:8]   # [4, 67, 12, 45, 23]   "to get very tired of"
```

### **Step 4C: Visual Representation**

```
Position:     0    1    2    3    4    5    6    7    8    9
Tokens:    [  8,   9, 234,   4,  67,  12,  45,  23, 156,  34 ]
Words:     alice was beginning to get very tired  of sitting by

Training Example 1:
Input:     [  8,   9, 234,   4,  67 ]  "alice was beginning to get"
Target:    [  9, 234,   4,  67,  12 ]  "was beginning to get very"
           ↑    ↑    ↑    ↑    ↑
          Predict these from the inputs above

Training Example 2:
Input:     [  9, 234,   4,  67,  12 ]  "was beginning to get very"  
Target:    [234,   4,  67,  12,  45 ]  "beginning to get very tired"
```

### **Step 4D: What Each Training Example Teaches**

Each position in the sequence learns a different prediction:

```python
input_seq  = [8, 9, 234, 4, 67]     # "alice was beginning to get"
target_seq = [9, 234, 4, 67, 12]    # "was beginning to get very"

# What the model learns:
# Position 0: Given "alice" → predict "was"
# Position 1: Given "alice was" → predict "beginning"  
# Position 2: Given "alice was beginning" → predict "to"
# Position 3: Given "alice was beginning to" → predict "get"
# Position 4: Given "alice was beginning to get" → predict "very"
```

### **Step 4E: Dataset Class Methods**

```python
def __len__(self):
    return len(self.examples)  # How many training examples we have

def __getitem__(self, idx):
    input_seq, target_seq = self.examples[idx]
    
    # Convert to PyTorch tensors (required format)
    input_tensor = torch.tensor(input_seq, dtype=torch.long)
    target_tensor = torch.tensor(target_seq, dtype=torch.long)
    
    return input_tensor, target_tensor
```

### **Step 4F: Complete Example with Real Numbers**

Let's say our tokenized text is:
```python
tokens = [8, 9, 234, 4, 67, 12, 45, 23, 156, 34, 89, 67, 234, 445, 23, 67, 89]
# Length: 17 tokens
# With seq_length = 5, we get: 17 - 5 = 12 training examples
```

**All training examples:**
```python
examples = [
    # (input_seq, target_seq)
    ([8, 9, 234, 4, 67], [9, 234, 4, 67, 12]),      # Example 0
    ([9, 234, 4, 67, 12], [234, 4, 67, 12, 45]),    # Example 1  
    ([234, 4, 67, 12, 45], [4, 67, 12, 45, 23]),    # Example 2
    ([4, 67, 12, 45, 23], [67, 12, 45, 23, 156]),   # Example 3
    ([67, 12, 45, 23, 156], [12, 45, 23, 156, 34]), # Example 4
    # ... and so on for 12 total examples
]
```

### **Step 4G: PyTorch Tensors (Data Format)**

```python
# When we call dataset[0], we get:
input_tensor = torch.tensor([8, 9, 234, 4, 67], dtype=torch.long)
target_tensor = torch.tensor([9, 234, 4, 67, 12], dtype=torch.long)

# Tensor properties:
print(input_tensor.shape)   # torch.Size([5])  - 1D tensor with 5 elements
print(input_tensor.dtype)   # torch.int64      - 64-bit integers
print(input_tensor.device)  # cpu              - stored on CPU (for now)
```

### **Step 4H: Memory Layout**

**In memory, each example looks like:**
```python
Example 0:
├── input_tensor:  [8, 9, 234, 4, 67]     # Shape: [5]
└── target_tensor: [9, 234, 4, 67, 12]    # Shape: [5]

Example 1:
├── input_tensor:  [9, 234, 4, 67, 12]    # Shape: [5]  
└── target_tensor: [234, 4, 67, 12, 45]   # Shape: [5]
```

### **Step 4I: Dataset Split (Train/Validation)**

```python
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size    # 20% for validation

train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)
```

**Why split the data?**
- **Training set**: Model learns from these examples
- **Validation set**: Test how well model generalizes to unseen data
- **Prevents overfitting**: Model memorizing instead of learning patterns

### **Step 4J: DataLoader (Batching)**

```python
batch_size = 8

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,      # Mix up the order each epoch
    drop_last=True     # Drop incomplete batches
)
```

**What batching does:**
Instead of processing one example at a time, we process 8 examples together:

```python
# Single example:
input_shape: [5]        # 5 tokens
target_shape: [5]       # 5 tokens

# Batch of 8 examples:
input_shape: [8, 5]     # 8 examples, each with 5 tokens  
target_shape: [8, 5]    # 8 targets, each with 5 tokens
```

**Batch visualization:**
```python
batch_input = [
    [8, 9, 234, 4, 67],      # Example 1
    [9, 234, 4, 67, 12],     # Example 2
    [234, 4, 67, 12, 45],    # Example 3
    [4, 67, 12, 45, 23],     # Example 4
    [67, 12, 45, 23, 156],   # Example 5
    [12, 45, 23, 156, 34],   # Example 6
    [45, 23, 156, 34, 89],   # Example 7
    [23, 156, 34, 89, 67]    # Example 8
]
# Shape: [8, 5] = [batch_size, seq_length]
```

### **Key Insights:**

1. **Sliding window creates many examples** from single text
2. **Each position learns different context lengths** (1 word, 2 words, 3 words, etc.)
3. **Target is always input shifted by 1** (next word prediction)
4. **Batching enables parallel processing** on GPU
5. **Train/val split prevents overfitting**

### **What's stored in memory:**
```python
dataset.examples = [
    ([8, 9, 234, 4, 67], [9, 234, 4, 67, 12]),
    ([9, 234, 4, 67, 12], [234, 4, 67, 12, 45]),
    # ... thousands of examples
]
```
---
## Step 5: Positional Encoding - Teaching the Model About Word Order

**The Problem:** Neural networks don't naturally understand that "cat sat mat" is different from "mat sat cat". They process all words at the same time!

**The Solution:** Add special position numbers to each word's embedding.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=1000):
        super().__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).float().unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        
        self.register_buffer('pe', pe.unsqueeze(0))
```

### **Step 5A: Understanding the Problem**

Without positional encoding:
```python
# These sentences would look IDENTICAL to the neural network:
sentence1 = ["the", "cat", "sat", "on", "mat"]
sentence2 = ["mat", "on", "sat", "cat", "the"]

# Because neural networks process them as a "bag of words"
# Missing: WHERE each word appears in the sentence
```

### **Step 5B: The Mathematical Formula**

For each position `pos` and dimension `i`:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))     # Even dimensions
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))     # Odd dimensions
```

**Breaking down the formula:**

```python
# Parameters:
pos = 0, 1, 2, 3, 4, ...        # Position in sentence (0=first word, 1=second, etc.)
d_model = 128                    # Embedding dimension
i = 0, 1, 2, 3, ..., d_model/2  # Dimension index
```

### **Step 5C: Step-by-Step Calculation**

Let's calculate position encoding for **position 0** (first word) and **d_model=4** (simplified):

```python
# Step 1: Create position tensor
position = torch.arange(0, max_length).float().unsqueeze(1)
# Result: [[0.], [1.], [2.], [3.], [4.], ...]  Shape: [max_length, 1]

# Step 2: Calculate division term
div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

# For d_model=4:
# torch.arange(0, 4, 2) = [0, 2]  # Even dimensions only
# -(math.log(10000.0) / 4) = -2.302
# torch.exp([0, 2] * -2.302) = torch.exp([0, -4.605]) = [1.0, 0.01]

div_term = [1.0, 0.01]
```

**Step 3: Calculate sine and cosine values**

For **position 0** (first word):
```python
pos = 0

# Even dimensions (0, 2):
pe[0, 0] = sin(0 * 1.0) = sin(0) = 0.0
pe[0, 2] = sin(0 * 0.01) = sin(0) = 0.0

# Odd dimensions (1, 3):
pe[0, 1] = cos(0 * 1.0) = cos(0) = 1.0  
pe[0, 3] = cos(0 * 0.01) = cos(0) = 1.0

# Position 0 encoding: [0.0, 1.0, 0.0, 1.0]
```

For **position 1** (second word):
```python
pos = 1

# Even dimensions:
pe[1, 0] = sin(1 * 1.0) = sin(1) = 0.841
pe[1, 2] = sin(1 * 0.01) = sin(0.01) = 0.01

# Odd dimensions:
pe[1, 1] = cos(1 * 1.0) = cos(1) = 0.540
pe[1, 3] = cos(1 * 0.01) = cos(0.01) = 0.9999

# Position 1 encoding: [0.841, 0.540, 0.01, 0.9999]
```

### **Step 5D: Complete Positional Encoding Matrix**

For a sequence length of 5 and d_model=4:

```python
pe = [
    [0.000, 1.000, 0.000, 1.000],  # Position 0
    [0.841, 0.540, 0.010, 0.999],  # Position 1  
    [0.909, -0.416, 0.020, 0.999], # Position 2
    [0.141, -0.989, 0.030, 0.999], # Position 3
    [-0.756, -0.654, 0.040, 0.999] # Position 4
]
# Shape: [5, 4] = [seq_length, d_model]
```

### **Step 5E: How Positional Encoding is Applied**

```python
def forward(self, x):
    seq_len = x.size(1)
    x = x + self.pe[:, :seq_len]  # Add position info to word embeddings
    return x
```

**Example application:**

```python
# Input: Word embeddings for "the cat sat"
word_embeddings = [
    [0.1, 0.2, 0.3, 0.4],  # "the" embedding
    [0.5, 0.6, 0.7, 0.8],  # "cat" embedding  
    [0.9, 1.0, 1.1, 1.2]   # "sat" embedding
]
# Shape: [3, 4] = [seq_length, d_model]

# Add positional encoding:
positional_encoding = [
    [0.000, 1.000, 0.000, 1.000],  # Position 0
    [0.841, 0.540, 0.010, 0.999],  # Position 1
    [0.909, -0.416, 0.020, 0.999]  # Position 2
]

# Final embeddings (word + position):
final_embeddings = [
    [0.1+0.000, 0.2+1.000, 0.3+0.000, 0.4+1.000] = [0.1, 1.2, 0.3, 1.4],
    [0.5+0.841, 0.6+0.540, 0.7+0.010, 0.8+0.999] = [1.341, 1.14, 0.71, 1.799],
    [0.9+0.909, 1.0-0.416, 1.1+0.020, 1.2+0.999] = [1.809, 0.584, 1.12, 2.199]
]
```

### **Step 5F: Why This Works**

**Key properties:**

1. **Unique fingerprint**: Each position gets a unique pattern
2. **Relative positions**: Model can learn "word A comes before word B"
3. **Periodic patterns**: Similar positions have similar encodings
4. **Scalable**: Works for any sequence length

### **Step 5G: Visual Understanding**

Imagine each position as a unique "barcode":

```
Position 0: ||||    |    ||||    |     (0.0, 1.0, 0.0, 1.0)
Position 1: |||| || |    |||| ||||     (0.841, 0.540, 0.01, 0.999)  
Position 2: ||||||||     ||||||||      (0.909, -0.416, 0.02, 0.999)
Position 3: |||  |||     ||||||||      (0.141, -0.989, 0.03, 0.999)
Position 4:  ||   ||     ||||||||      (-0.756, -0.654, 0.04, 0.999)
```

### **Step 5H: Memory Storage**

```python
# What gets stored in memory:
self.pe = torch.tensor([
    [[0.000, 1.000, 0.000, 1.000, ...],  # Position 0
     [0.841, 0.540, 0.010, 0.999, ...],  # Position 1
     [0.909, -0.416, 0.020, 0.999, ...], # Position 2
     ...                                  # Up to max_length positions
     [pos_n_encoding...]]                 # Position max_length-1
])
# Shape: [1, max_length, d_model] = [1, 1000, 128]
```

### **Step 5I: The Unsqueeze Operation**

```python
pe = torch.zeros(max_length, d_model)        # Shape: [1000, 128]
self.register_buffer('pe', pe.unsqueeze(0))  # Shape: [1, 1000, 128]
```

**Why unsqueeze(0)?**
- Adds batch dimension for broadcasting
- Allows same positional encoding to be applied to all examples in a batch

### **Step 5J: Real-World Example**

For our model with d_model=128 and seq_length=24:

```python
# Sentence: "alice was beginning to get very tired"
# Positions: [0, 1, 2, 3, 4, 5, 6]

# Each word gets word_embedding + position_encoding:
alice_final = alice_embedding + position_0_encoding    # [128] + [128] = [128]
was_final = was_embedding + position_1_encoding        # [128] + [128] = [128]  
beginning_final = beginning_embedding + position_2_encoding  # [128] + [128] = [128]
# ... and so on
```

### **Key Insights:**

1. **Position encoding is learned from data**: The specific values come from math, not training
2. **Same word, different positions**: "was" at position 1 vs position 5 will have different final embeddings
3. **Preserves meaning**: Original word meaning + position information
4. **No extra parameters**: Just mathematical computation, no weights to learn

### **What happens next:**
The model now knows that "cat sat mat" ≠ "mat sat cat" because each word has position information encoded into it!

---

## Step 6: Multi-Head Attention - The Heart of the Transformer

This is the **most important part**! Attention lets the model decide which words to focus on when predicting the next word.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model          # 128 (total embedding size)
        self.num_heads = num_heads      # 8 (number of attention heads)
        self.d_k = d_model // num_heads # 16 (size per head: 128/8=16)
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # Query projection
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # Key projection
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # Value projection
        self.W_o = nn.Linear(d_model, d_model)              # Output projection
```

### **Step 6A: The Attention Concept**

**Real-world analogy:** When reading "The cat sat on the mat", to understand "sat", you need to pay attention to:
- **"cat"** (what is sitting?)
- **"on"** (where is it sitting?)
- **"mat"** (what is it sitting on?)

**In neural networks:** Attention computes how much each word should influence the understanding of every other word.

### **Step 6B: Query, Key, Value (QKV) Concept**

Think of attention like a **search engine**:

- **Query (Q)**: "What am I looking for?" 
- **Key (K)**: "What do I have to offer?"
- **Value (V)**: "What information do I contain?"

**Example:**
```python
sentence = "the cat sat on the mat"

# When processing "sat":
query = "what_action_is_happening?"
keys = ["article", "animal", "action", "preposition", "article", "object"]  
values = ["the_info", "cat_info", "sat_info", "on_info", "the_info", "mat_info"]

# Attention finds: "cat" and "mat" are most relevant for understanding "sat"
```

### **Step 6C: Mathematical Foundation**

The core attention formula:
```
Attention(Q,K,V) = softmax(QK^T / √d_k)V
```

Let's break this down step by step:

### **Step 6D: Creating Q, K, V Matrices**

```python
def forward(self, query, key, value, mask=None):
    batch_size, seq_length = query.size(0), query.size(1)
    
    Q = self.W_q(query)  # [batch, seq_len, d_model] → [batch, seq_len, d_model]
    K = self.W_k(key)    # [batch, seq_len, d_model] → [batch, seq_len, d_model]
    V = self.W_v(value)  # [batch, seq_len, d_model] → [batch, seq_len, d_model]
```

**Example with real numbers:**
```python
# Input embeddings for "the cat sat" (simplified to d_model=4)
input_embeddings = [
    [0.1, 0.2, 0.3, 0.4],  # "the" with position encoding
    [0.5, 0.6, 0.7, 0.8],  # "cat" with position encoding
    [0.9, 1.0, 1.1, 1.2]   # "sat" with position encoding
]
# Shape: [1, 3, 4] = [batch_size, seq_length, d_model]

# Linear transformations (W_q, W_k, W_v are learned weight matrices):
W_q = [[0.1, 0.2, 0.3, 0.4],
       [0.2, 0.3, 0.4, 0.5],
       [0.3, 0.4, 0.5, 0.6],
       [0.4, 0.5, 0.6, 0.7]]  # Shape: [4, 4]

# Q = input_embeddings @ W_q
Q = [[0.3, 0.4, 0.5, 0.6],   # Query for "the"
     [0.7, 0.8, 0.9, 1.0],   # Query for "cat"  
     [1.1, 1.2, 1.3, 1.4]]   # Query for "sat"
# Shape: [1, 3, 4]

# K and V are computed similarly with W_k and W_v
```

### **Step 6E: Multi-Head Split**

Instead of one big attention, we split into multiple "heads":

```python
# Reshape for multi-head attention:
Q = Q.view(batch_size, seq_length, num_heads, d_k).transpose(1, 2)
K = K.view(batch_size, seq_length, num_heads, d_k).transpose(1, 2)
V = V.view(batch_size, seq_length, num_heads, d_k).transpose(1, 2)
```

**Visual representation:**
```python
# Before reshaping:
Q.shape = [1, 3, 8]  # [batch, seq_len, d_model] where d_model=8, num_heads=4, d_k=2

Q = [[Q_word1_all8dims],
     [Q_word2_all8dims], 
     [Q_word3_all8dims]]

# After reshaping and transpose:
Q.shape = [1, 4, 3, 2]  # [batch, num_heads, seq_len, d_k]

Q = [[[Q_word1_head1], [Q_word2_head1], [Q_word3_head1]],  # Head 1
     [[Q_word1_head2], [Q_word2_head2], [Q_word3_head2]],  # Head 2
     [[Q_word1_head3], [Q_word2_head3], [Q_word3_head3]],  # Head 3
     [[Q_word1_head4], [Q_word2_head4], [Q_word3_head4]]]  # Head 4
```

### **Step 6F: Scaled Dot-Product Attention**

```python
def scaled_dot_product_attention(self, Q, K, V, mask=None):
    # Step 1: Calculate attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
    
    # Step 2: Apply mask (prevent looking at future words)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Step 3: Convert to probabilities
    attention_weights = F.softmax(scores, dim=-1)
    
    # Step 4: Apply attention to values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
```

### **Step 6G: Step-by-Step Attention Calculation**

Let's compute attention for "the cat sat" with simplified numbers:

**Step 1: Compute scores (QK^T)**
```python
# For one head, simplified to d_k=2:
Q = [[0.1, 0.2],   # Query for "the"
     [0.3, 0.4],   # Query for "cat"
     [0.5, 0.6]]   # Query for "sat"

K = [[0.2, 0.1],   # Key for "the"  
     [0.4, 0.3],   # Key for "cat"
     [0.6, 0.5]]   # Key for "sat"

# Matrix multiplication QK^T:
scores = Q @ K.T = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]] @ [[0.2, 0.4, 0.6],
                                                              [0.1, 0.3, 0.5]]

scores = [[0.04, 0.10, 0.16],   # How much "the" attends to [the, cat, sat]
          [0.10, 0.21, 0.38],   # How much "cat" attends to [the, cat, sat]  
          [0.16, 0.38, 0.60]]   # How much "sat" attends to [the, cat, sat]
```

**Step 2: Scale by √d_k**
```python
scores = scores / math.sqrt(2) = scores / 1.414

scores = [[0.028, 0.071, 0.113],
          [0.071, 0.148, 0.269], 
          [0.113, 0.269, 0.424]]
```

**Step 3: Apply causal mask**
```python
# Causal mask (prevent looking at future words):
mask = [[1, 0, 0],    # "the" can only see "the"
        [1, 1, 0],    # "cat" can see "the, cat"  
        [1, 1, 1]]    # "sat" can see "the, cat, sat"

# Apply mask (set forbidden positions to -∞):
masked_scores = [[0.028,  -∞,    -∞  ],
                 [0.071, 0.148,  -∞  ],
                 [0.113, 0.269, 0.424]]
```

**Step 4: Softmax to get probabilities**
```python
attention_weights = softmax(masked_scores)

# For each row, probabilities sum to 1:
attention_weights = [[1.0,   0.0,   0.0 ],   # "the" pays 100% attention to itself
                     [0.481, 0.519, 0.0 ],   # "cat" pays 48% to "the", 52% to itself
                     [0.307, 0.340, 0.353]]  # "sat" pays attention to all three words
```

**Step 5: Apply attention to values**
```python
V = [[0.1, 0.3],   # Value for "the"
     [0.2, 0.4],   # Value for "cat"  
     [0.5, 0.7]]   # Value for "sat"

# Weighted combination:
output = attention_weights @ V

# For "the": 1.0*[0.1,0.3] + 0.0*[0.2,0.4] + 0.0*[0.5,0.7] = [0.1, 0.3]
# For "cat": 0.481*[0.1,0.3] + 0.519*[0.2,0.4] + 0.0*[0.5,0.7] = [0.152, 0.351]
# For "sat": 0.307*[0.1,0.3] + 0.340*[0.2,0.4] + 0.353*[0.5,0.7] = [0.276, 0.565]

output = [[0.1,   0.3  ],    # Updated representation for "the"
          [0.152, 0.351],    # Updated representation for "cat"
          [0.276, 0.565]]    # Updated representation for "sat"
```

### **Step 6H: Multi-Head Combination**

```python
# After all heads compute their outputs:
head_outputs = [
    output_head_1,  # [batch, seq_len, d_k]
    output_head_2,  # [batch, seq_len, d_k]
    # ... 8 heads total
]

# Concatenate all heads:
attn_output = torch.cat(head_outputs, dim=-1)  # [batch, seq_len, d_model]

# Final linear transformation:
output = self.W_o(attn_output)
```

### **Step 6I: Why Multiple Heads?**

Each head can learn different types of relationships:

```python
# Example with "The cat sat on the mat":

# Head 1: Subject-Verb relationships
# "cat" → "sat" (who is doing the action?)

# Head 2: Spatial relationships  
# "sat" → "on" → "mat" (where is the action happening?)

# Head 3: Article-Noun relationships
# "the" → "cat", "the" → "mat" (which specific objects?)

# Head 4: Sequential relationships
# Each word → previous word (word order patterns)
```

### **Step 6J: Memory Layout**

```python
# What's stored for each attention head:
attention_weights = [
    # Head 1:
    [[1.0,   0.0,   0.0,   0.0,   0.0],    # Word 1 attention distribution
     [0.3,   0.7,   0.0,   0.0,   0.0],    # Word 2 attention distribution
     [0.1,   0.4,   0.5,   0.0,   0.0],    # Word 3 attention distribution
     [0.2,   0.2,   0.3,   0.3,   0.0],    # Word 4 attention distribution
     [0.1,   0.2,   0.2,   0.3,   0.2]],   # Word 5 attention distribution
    
    # Head 2:
    [[1.0,   0.0,   0.0,   0.0,   0.0],
     [0.8,   0.2,   0.0,   0.0,   0.0],
     # ... different attention pattern
    ],
    # ... 8 heads total
]
# Shape: [num_heads, seq_len, seq_len] = [8, 5, 5]
```

### **Step 6K: Causal Mask (Critical for Language Modeling)**

```python
def create_causal_mask(seq_length):
    mask = torch.tril(torch.ones(seq_length, seq_length))
    return mask.unsqueeze(0).unsqueeze(0)

# For seq_length=5:
mask = [[1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0], 
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1]]
```

**Why causal mask?**
- Prevents model from "cheating" by looking at future words
- Word at position i can only attend to positions 0, 1, ..., i
- Essential for autoregressive generation (predicting next word)

### **Key Insights:**

1. **Attention = Weighted Average**: Each word becomes a weighted combination of all previous words
2. **Learning What Matters**: Model learns which words are important for understanding each position
3. **Context-Dependent**: Same word gets different representations based on context
4. **Parallel Processing**: All positions computed simultaneously (unlike RNNs)
5. **Multiple Perspectives**: Each head learns different types of relationships

### **The Magic:**
After attention, "sat" doesn't just mean "sat" - it means "the cat's action of sitting" because it has been enriched with information from "the" and "cat"!

---

## Step 7: Feed Forward Network - Processing the Attended Information

After attention tells us WHAT to focus on, the feed forward network decides WHAT TO DO with that information.

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model  # 128 (input/output dimension)
        self.d_ff = d_ff        # 512 (hidden dimension - 4x larger!)
        
        self.linear1 = nn.Linear(d_model, d_ff)    # Expand: 128 → 512
        self.linear2 = nn.Linear(d_ff, d_model)    # Compress: 512 → 128
        self.dropout = nn.Dropout(dropout)
```

### **Step 7A: The Concept - Think Then Decide**

**Real-world analogy:** After paying attention to relevant words, you need to "think" about what they mean together.

```python
# After attention: "sat" now contains information about "cat" and "mat"
attended_sat = "cat's_action_of_sitting_on_mat"

# Feed forward network thinks:
# "Hmm, I see a cat + sitting + mat... this suggests a resting action on furniture"
# Then outputs: enhanced_understanding_of_sat
```

### **Step 7B: The Architecture - Expand, Activate, Compress**

```python
def forward(self, x):
    # Step 1: Expand to larger dimension (more "thinking space")
    x = self.linear1(x)      # [batch, seq_len, 128] → [batch, seq_len, 512]
    
    # Step 2: Apply ReLU activation (non-linearity)
    x = F.relu(x)           # Remove negative values, keep positive
    
    # Step 3: Apply dropout (prevent overfitting)
    x = self.dropout(x)
    
    # Step 4: Compress back to original dimension
    x = self.linear2(x)     # [batch, seq_len, 512] → [batch, seq_len, 128]
    
    return x
```

### **Step 7C: Why 4x Expansion?**

```python
d_model = 128    # Input dimension
d_ff = 512       # Hidden dimension (4x larger)

# Think of it like this:
# Input: "I have 128 pieces of information about this word"
# Expansion: "Let me spread this into 512 thinking slots"
# Processing: "Now I can do complex reasoning in this larger space"
# Compression: "Summarize my thoughts back into 128 pieces"
```

**Why larger dimension helps:**
- More parameters = more complex patterns
- More "thinking space" for the model
- Can combine information in sophisticated ways

### **Step 7D: Step-by-Step Example**

Let's process the attended representation of "sat":

```python
# Input: attended "sat" representation
input_sat = [0.3, 0.5, -0.2, 0.8]  # Simplified to 4 dimensions

# Step 1: Linear expansion (128 → 512, simplified to 4 → 8)
W1 = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
      [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
      [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
      [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]]

expanded = input_sat @ W1 = [0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6]

# Step 2: ReLU activation (remove negatives)
activated = relu(expanded) = [0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6]
# (All values were positive, so no change)

# Step 3: Dropout (randomly set some to 0 during training)
after_dropout = [0.5, 0.0, 1.1, 1.4, 0.0, 2.0, 2.3, 0.0]  # Random example

# Step 4: Linear compression (8 → 4)
W2 = [[0.1, 0.2, 0.3, 0.4],
      [0.2, 0.3, 0.4, 0.5],
      [0.3, 0.4, 0.5, 0.6],
      [0.4, 0.5, 0.6, 0.7],
      [0.5, 0.6, 0.7, 0.8],
      [0.6, 0.7, 0.8, 0.9],
      [0.7, 0.8, 0.9, 1.0],
      [0.8, 0.9, 1.0, 1.1]]

final_output = after_dropout @ W2 = [1.2, 1.5, 1.8, 2.1]
```

### **Step 7E: What Feed Forward Actually Learns**

The feed forward network learns **feature combinations**:

```python
# Example patterns the network might learn:

# Pattern 1: "Action + Object" detector
if expanded[0] > 0.5 and expanded[3] > 0.8:
    # This might indicate "action happening to object"
    output[0] = high_value

# Pattern 2: "Spatial relationship" detector  
if expanded[1] > 0.6 and expanded[4] > 0.7:
    # This might indicate "spatial positioning"
    output[1] = high_value

# Pattern 3: "Temporal sequence" detector
if expanded[2] > 0.4 and expanded[5] > 0.9:
    # This might indicate "time-based action"
    output[2] = high_value
```

### **Step 7F: ReLU Activation - Why It Matters**

```python
# Without ReLU (just linear transformations):
# Model can only learn linear relationships
# Example: output = 2*input + 3

# With ReLU:
# Model can learn complex, non-linear patterns
# Example: if input > threshold then activate_pattern_A else activate_pattern_B

def relu_example():
    values = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    after_relu = [max(0, x) for x in values]
    # Result: [0.0, 0.0, 0.0, 0.5, 1.0, 1.5]
    
    # Effect: Creates "gates" - some neurons turn off (0), others stay on
```

### **Step 7G: Memory Layout**

```python
# What gets stored in each layer:

# Linear1 weights: [d_model, d_ff] = [128, 512]
W1 = torch.tensor([
    [w1_00, w1_01, w1_02, ..., w1_0511],  # How input dim 0 connects to all 512 hidden dims
    [w1_10, w1_11, w1_12, ..., w1_1511],  # How input dim 1 connects to all 512 hidden dims
    # ... 128 rows total
])

# Linear2 weights: [d_ff, d_model] = [512, 128]  
W2 = torch.tensor([
    [w2_00, w2_01, w2_02, ..., w2_0127],  # How hidden dim 0 connects to all 128 output dims
    [w2_10, w2_11, w2_12, ..., w2_1127],  # How hidden dim 1 connects to all 128 output dims
    # ... 512 rows total
])
```

### **Step 7H: Position-wise Processing**

**Key insight:** Feed forward processes each position independently!

```python
sentence = "the cat sat on mat"
positions = [pos_0, pos_1, pos_2, pos_3, pos_4]

# Each position goes through the SAME feed forward network:
for position in positions:
    enhanced_position = feed_forward(position)

# This means:
# - "cat" at position 1 gets the same processing as "mat" at position 4
# - But their inputs are different (due to attention), so outputs differ
# - No information flows between positions in feed forward
```

### **Step 7I: Parameter Count**

```python
# Feed forward parameters:
linear1_params = d_model * d_ff = 128 * 512 = 65,536
linear2_params = d_ff * d_model = 512 * 128 = 65,536
total_ff_params = 131,072

# This is typically the LARGEST component of the transformer!
# Much bigger than attention: ~131K vs ~65K parameters
```

### **Step 7J: What Happens in Practice**

```python
# Input: Attended representations
input_batch = [
    [[0.1, 0.3, -0.2, 0.5, ...],  # "the" (128 dims)
     [0.4, 0.6, 0.1, 0.8, ...],   # "cat" (128 dims)
     [0.2, 0.9, -0.1, 0.3, ...]   # "sat" (128 dims)
    ],
    # ... more examples in batch
]

# After feed forward:
output_batch = [
    [[0.2, 0.4, 0.1, 0.7, ...],   # Enhanced "the"
     [0.3, 0.8, 0.2, 0.9, ...],   # Enhanced "cat"  
     [0.5, 0.6, 0.3, 0.4, ...]    # Enhanced "sat"
    ],
    # ... enhanced representations
]

# Each word now has:
# 1. Original word meaning (from embedding)
# 2. Position information (from positional encoding)
# 3. Context from other words (from attention)
# 4. Complex feature combinations (from feed forward)
```

### **Key Insights:**

1. **Expansion and compression**: Gives model more "thinking space"
2. **Non-linearity**: ReLU enables learning complex patterns
3. **Position-wise**: Each word processed independently
4. **Feature combination**: Learns to combine attended information
5. **Most parameters**: Usually 2/3 of transformer's parameters

### **The Role in the Big Picture:**
- **Attention says**: "Focus on these words"
- **Feed Forward says**: "Now that I'm focused, here's what it all means"

---

## Step 8: Transformer Block - Combining Everything with Crucial Tricks

This is where we combine attention + feed forward + some **essential tricks** that make training actually work!

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # CRUCIAL COMPONENTS:
        self.norm1 = nn.LayerNorm(d_model)  # After attention
        self.norm2 = nn.LayerNorm(d_model)  # After feed forward
        self.dropout = nn.Dropout(dropout)
```

### **Step 8A: The Two Essential Tricks**

**Trick 1: Residual Connections** (Skip connections)
**Trick 2: Layer Normalization**

Without these, deep transformers **DON'T WORK AT ALL!**

### **Step 8B: Residual Connections - The Highway for Information**

```python
def forward(self, x, mask=None):
    # Attention with residual connection
    attn_output, attention_weights = self.attention(
        self.norm1(x), self.norm1(x), self.norm1(x), mask
    )
    x = x + self.dropout(attn_output)  # ← THIS IS THE RESIDUAL CONNECTION
    
    # Feed forward with residual connection  
    ff_output = self.feed_forward(self.norm2(x))
    x = x + self.dropout(ff_output)    # ← THIS IS THE RESIDUAL CONNECTION
    
    return x, attention_weights
```

### **Step 8C: Why Residual Connections Are Essential**

**The Problem:** Deep networks suffer from "vanishing gradients"

```python
# Without residual connections (BAD):
x = input_embedding              # [0.1, 0.2, 0.3, 0.4]
x = attention_layer(x)           # [0.05, 0.08, 0.12, 0.15] (getting smaller)
x = feed_forward_layer(x)        # [0.02, 0.03, 0.04, 0.05] (even smaller)
x = another_attention_layer(x)   # [0.008, 0.01, 0.015, 0.02] (almost zero!)
# After many layers: [0.0001, 0.0002, 0.0003, 0.0004] (information is lost!)

# With residual connections (GOOD):
x = input_embedding              # [0.1, 0.2, 0.3, 0.4]
x = x + attention_layer(x)       # [0.1, 0.2, 0.3, 0.4] + [small_changes] = preserved!
x = x + feed_forward_layer(x)    # Original info + new info = still rich!
# Information never disappears!
```

### **Step 8D: Layer Normalization - Keeping Numbers Stable**

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))   # Learnable scale
        self.beta = nn.Parameter(torch.zeros(d_model))   # Learnable shift
        self.eps = eps
    
    def forward(self, x):
        # Calculate mean and variance across the last dimension
        mean = x.mean(dim=-1, keepdim=True)              # Average of each embedding
        var = x.var(dim=-1, keepdim=True)                # Variance of each embedding
        
        # Normalize: subtract mean, divide by standard deviation
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift (learnable parameters)
        return self.gamma * normalized + self.beta
```

### **Step 8E: Layer Norm Step-by-Step Example**

```python
# Input: one word's embedding after attention
x = [2.0, 8.0, 1.0, 5.0]  # Unbalanced values!

# Step 1: Calculate statistics
mean = (2.0 + 8.0 + 1.0 + 5.0) / 4 = 4.0
variance = ((2-4)² + (8-4)² + (1-4)² + (5-4)²) / 4 = (4 + 16 + 9 + 1) / 4 = 7.5
std_dev = sqrt(7.5) = 2.74

# Step 2: Normalize (mean=0, std=1)
normalized = [(2.0-4.0)/2.74, (8.0-4.0)/2.74, (1.0-4.0)/2.74, (5.0-4.0)/2.74]
normalized = [-0.73, 1.46, -1.09, 0.36]

# Step 3: Apply learnable parameters
gamma = [1.0, 1.0, 1.0, 1.0]  # (learned during training)
beta = [0.0, 0.0, 0.0, 0.0]   # (learned during training)

final = gamma * normalized + beta = [-0.73, 1.46, -1.09, 0.36]
```

### **Step 8F: Why Layer Norm Helps**

**Problem:** Neural networks are sensitive to input scale

```python
# Without layer norm:
word1 = [0.1, 0.2, 0.1, 0.15]     # Small values
word2 = [10.0, 20.0, 15.0, 25.0]  # Large values

# Network treats these VERY differently, even if they represent similar concepts!

# With layer norm:
word1_normalized = [-0.8, 0.8, -0.8, 0.0]   # Standardized scale
word2_normalized = [-0.9, 0.9, -0.3, 1.2]   # Same scale range

# Now network can focus on patterns, not magnitudes!
```

### **Step 8G: Pre-Norm vs Post-Norm**

Our implementation uses **Pre-Norm** (normalize first, then apply layer):

```python
# Pre-Norm (what we use - MORE STABLE):
attn_output = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
x = x + attn_output

# Post-Norm (older approach - LESS STABLE):
attn_output = self.attention(x, x, x, mask)  
x = self.norm1(x + attn_output)
```

**Why Pre-Norm is better:**
- More stable gradients
- Easier to train deep models
- Less likely to explode or vanish

### **Step 8H: Complete Forward Pass Example**

```python
# Input: word embeddings with position encoding
input_x = [
    [0.5, 0.3, 0.8, 0.2],  # "the"
    [0.1, 0.9, 0.4, 0.7],  # "cat" 
    [0.6, 0.2, 0.1, 0.8]   # "sat"
]

# Step 1: Layer norm before attention
normed_x = layer_norm(input_x)
# Result: normalized versions of each embedding

# Step 2: Multi-head attention
attn_output, weights = attention(normed_x, normed_x, normed_x, mask)
# Result: contextual information for each word

# Step 3: Residual connection + dropout
x = input_x + dropout(attn_output)
# Result: original info + attention info

# Step 4: Layer norm before feed forward  
normed_x2 = layer_norm(x)

# Step 5: Feed forward network
ff_output = feed_forward(normed_x2) 

# Step 6: Second residual connection + dropout
final_x = x + dropout(ff_output)
# Result: original + attention + feed forward info

# Final output: Each word now has rich, multi-layered representation
```

### **Step 8I: Information Flow Visualization**

```python
# What each word contains at each step:

# Initial: 
"cat" = word_embedding + position_encoding

# After attention:
"cat" = word_embedding + position_encoding + attention_to_context

# After feed forward:
"cat" = word_embedding + position_encoding + attention_to_context + complex_features

# Each layer ADDS information, never replaces it!
```

### **Step 8J: Why This Architecture Works**

**1. Information Preservation:**
```python
# Residual connections ensure no information is lost
original_meaning + contextual_info + processed_features = rich_representation
```

**2. Stable Training:**
```python
# Layer norm keeps values in good range for learning
no_explosion + no_vanishing = successful_training
```

**3. Parallel Processing:**
```python
# All positions processed simultaneously
fast_computation + gpu_efficient = scalable_model
```

### **Step 8K: Memory Layout**

```python
# What gets stored for a transformer block:

# Layer Norm 1 parameters:
norm1_gamma = [learnable_scale_factor_for_each_dim]  # Shape: [128]
norm1_beta = [learnable_bias_for_each_dim]           # Shape: [128]

# Attention parameters:
attention_weights = {
    'W_q': [128, 128],  # Query projection
    'W_k': [128, 128],  # Key projection  
    'W_v': [128, 128],  # Value projection
    'W_o': [128, 128]   # Output projection
}

# Layer Norm 2 parameters:
norm2_gamma = [learnable_scale_factor_for_each_dim]  # Shape: [128]
norm2_beta = [learnable_bias_for_each_dim]           # Shape: [128]

# Feed Forward parameters:
ff_weights = {
    'linear1': [128, 512],  # Expansion
    'linear2': [512, 128]   # Compression
}

# Total parameters per block: ~280K parameters
```

### **Step 8L: Multiple Blocks Create Depth**

```python
# GPT-style model stacks multiple transformer blocks:

input_embeddings
    ↓
TransformerBlock_1  # Learn basic patterns
    ↓  
TransformerBlock_2  # Learn more complex patterns
    ↓
TransformerBlock_3  # Learn even more complex patterns
    ↓
TransformerBlock_4  # Learn very sophisticated patterns
    ↓
output_predictions

# Each layer builds on the previous layer's understanding
```

### **Key Insights:**

1. **Residual connections** = Information highway (nothing gets lost)
2. **Layer normalization** = Keeps training stable
3. **Pre-norm** = Better for deep models
4. **Additive nature** = Each layer enriches representation
5. **Parallel processing** = All positions computed together

### **The Magic:**
After going through a transformer block, each word doesn't just know about itself - it knows about its context, its relationships, and complex patterns, while still retaining its original meaning!

---

## Step 9: Complete GPT Model - Putting It All Together

Now we combine everything into a complete language model that can generate text!

```python
class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 1. Convert token IDs to dense vectors
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Add position information  
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)
        
        # 3. Stack multiple transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 4. Final processing
        self.ln_final = nn.LayerNorm(d_model)
        
        # 5. Convert back to vocabulary probabilities
        self.output_head = nn.Linear(d_model, vocab_size)
```

### **Step 9A: The Complete Information Flow**

```python
# Input: Token IDs
input_ids = [8, 9, 234, 4, 67]  # "alice was beginning to get"

# Step 1: Token Embedding
token_embeddings = [
    [0.1, 0.2, 0.3, ..., 0.5],  # "alice" → 128-dim vector
    [0.4, 0.3, 0.8, ..., 0.2],  # "was" → 128-dim vector
    [0.2, 0.9, 0.1, ..., 0.7],  # "beginning" → 128-dim vector
    [0.6, 0.1, 0.4, ..., 0.3],  # "to" → 128-dim vector
    [0.3, 0.7, 0.2, ..., 0.9]   # "get" → 128-dim vector
]
# Shape: [1, 5, 128] = [batch_size, seq_length, d_model]

# Step 2: Add positional encoding
x = token_embeddings + positional_encoding
# Each word now knows WHAT it is and WHERE it is

# Step 3: Pass through transformer blocks
for transformer_block in self.transformer_blocks:
    x, attention_weights = transformer_block(x, mask)
# Each layer adds more sophisticated understanding

# Step 4: Final layer normalization
x = self.ln_final(x)

# Step 5: Convert to vocabulary predictions
logits = self.output_head(x)  # [1, 5, 800] = [batch, seq_len, vocab_size]
```

### **Step 9B: Token Embedding - Converting Numbers to Vectors**

```python
# Embedding lookup table:
token_embedding = nn.Embedding(vocab_size=800, d_model=128)

# What this creates:
embedding_table = [
    [0.1, 0.2, 0.3, ..., 0.8],  # Embedding for token 0 (<PAD>)
    [0.4, 0.1, 0.9, ..., 0.2],  # Embedding for token 1 (<UNK>)
    [0.7, 0.3, 0.1, ..., 0.5],  # Embedding for token 2 (<BOS>)
    # ... 800 rows total, one for each word in vocabulary
    [0.2, 0.8, 0.4, ..., 0.1]   # Embedding for token 799
]
# Shape: [800, 128]

# Lookup process:
input_ids = [8, 9, 234]  # "alice was beginning"
embeddings = [
    embedding_table[8],    # Get row 8 for "alice"
    embedding_table[9],    # Get row 9 for "was"  
    embedding_table[234]   # Get row 234 for "beginning"
]
```

### **Step 9C: The Causal Mask - Preventing Cheating**

```python
def create_causal_mask(seq_length):
    mask = torch.tril(torch.ones(seq_length, seq_length))
    return mask.unsqueeze(0).unsqueeze(0)

# For "alice was beginning to get":
mask = [
    [1, 0, 0, 0, 0],  # "alice" can only see "alice"
    [1, 1, 0, 0, 0],  # "was" can see "alice, was"
    [1, 1, 1, 0, 0],  # "beginning" can see "alice, was, beginning"  
    [1, 1, 1, 1, 0],  # "to" can see "alice, was, beginning, to"
    [1, 1, 1, 1, 1]   # "get" can see all previous words
]
```

**Why this is crucial:**
```python
# Without causal mask (CHEATING):
# When predicting what comes after "alice was", model can see "beginning to get"
# This makes training useless - model learns to copy, not predict!

# With causal mask (PROPER TRAINING):
# When predicting what comes after "alice was", model can only see "alice was"
# Model must actually learn language patterns!
```

### **Step 9D: Output Head - Converting to Predictions**

```python
# After all transformer blocks:
final_representations = [
    [0.3, 0.7, 0.1, ..., 0.9],  # Rich representation of "alice"
    [0.8, 0.2, 0.5, ..., 0.1],  # Rich representation of "was"
    [0.1, 0.9, 0.3, ..., 0.6],  # Rich representation of "beginning"
    [0.4, 0.1, 0.8, ..., 0.2],  # Rich representation of "to"
    [0.6, 0.3, 0.1, ..., 0.7]   # Rich representation of "get"
]
# Shape: [1, 5, 128]

# Output head: Linear layer [128 → 800]
logits = self.output_head(final_representations)

# Result: Predictions for each position
logits = [
    [2.3, 0.1, 0.8, ..., 1.2],  # Predictions for position 0 (after "alice")
    [0.5, 3.1, 0.2, ..., 0.9],  # Predictions for position 1 (after "was")  
    [1.1, 0.4, 2.8, ..., 0.3],  # Predictions for position 2 (after "beginning")
    [0.7, 1.9, 0.1, ..., 2.5],  # Predictions for position 3 (after "to")
    [2.1, 0.3, 1.4, ..., 0.6]   # Predictions for position 4 (after "get")
]
# Shape: [1, 5, 800] - Each position predicts over full vocabulary
```

### **Step 9E: Training Target Alignment**

```python
# Input sequence:    [8, 9, 234, 4, 67]     "alice was beginning to get"
# Target sequence:   [9, 234, 4, 67, 12]    "was beginning to get very"

# Training alignment:
# Position 0: Input="alice"     Target="was"        → Learn: alice → was
# Position 1: Input="was"       Target="beginning"  → Learn: was → beginning  
# Position 2: Input="beginning" Target="to"         → Learn: beginning → to
# Position 3: Input="to"        Target="get"        → Learn: to → get
# Position 4: Input="get"       Target="very"       → Learn: get → very
```

### **Step 9F: Loss Calculation**

```python
def forward(self, input_ids, targets=None):
    # ... (all the processing steps)
    
    loss = None
    if targets is not None:
        # Reshape for cross-entropy loss
        logits_flat = logits.view(-1, self.vocab_size)  # [batch*seq_len, vocab_size]
        targets_flat = targets.view(-1)                  # [batch*seq_len]
        
        # Calculate cross-entropy loss
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-1)
    
    return logits, loss, attention_maps
```

**Cross-entropy loss explained:**
```python
# For one prediction:
predicted_logits = [2.3, 0.1, 0.8, 1.2, ...]  # Raw scores for each word
target_word_id = 9                              # Correct answer is word 9

# Convert logits to probabilities
probabilities = softmax(predicted_logits)       # [0.82, 0.01, 0.03, 0.05, ...]

# Loss = -log(probability of correct word)
loss = -log(probabilities[9])

# If model predicts correctly: probability[9] = 0.9 → loss = -log(0.9) = 0.1 (low)
# If model predicts wrong: probability[9] = 0.1 → loss = -log(0.1) = 2.3 (high)
```

### **Step 9G: Weight Initialization - Starting Smart**

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        torch.nn.init.zeros_(module.bias)
        torch.nn.init.ones_(module.weight)
```

**Why careful initialization matters:**
```python
# Bad initialization (random large values):
weights = [[-5.0, 8.2, -3.1, 9.7], ...]
# → Exploding gradients, unstable training

# Good initialization (small random values):  
weights = [[0.02, -0.01, 0.03, -0.02], ...]
# → Stable gradients, successful training
```

### **Step 9H: Model Configuration Example**

```python
config = {
    'vocab_size': 800,        # Number of unique words
    'd_model': 128,           # Embedding dimension
    'num_heads': 8,           # Attention heads (128/8 = 16 dims per head)
    'num_layers': 4,          # Transformer blocks
    'd_ff': 512,              # Feed forward hidden dimension (4x d_model)
    'max_seq_length': 256,    # Maximum sequence length
    'dropout': 0.1            # Dropout rate
}

# Parameter count:
# Embeddings: 800 × 128 = 102,400
# 4 Transformer blocks: 4 × ~280,000 = 1,120,000  
# Output head: 128 × 800 = 102,400
# Total: ~1.3M parameters
```

### **Step 9I: Generation Process**

```python
def generate_text(model, tokenizer, prompt="alice", max_length=20):
    model.eval()
    
    # Start with prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)])  # [1, prompt_length]
    
    for _ in range(max_length):
        # Forward pass
        logits, _, _ = model(input_ids)  # [1, current_length, vocab_size]
        
        # Get predictions for last position only
        next_token_logits = logits[0, -1, :]  # [vocab_size]
        
        # Convert to probabilities
        probs = F.softmax(next_token_logits, dim=-1)
        
        # Sample next token  
        next_token = torch.multinomial(probs, num_samples=1)  # [1]
        
        # Add to sequence
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    # Decode back to text
    return tokenizer.decode(input_ids[0].tolist())
```

### **Step 9J: Memory Layout During Forward Pass**

```python
# Forward pass memory usage:

batch_size = 2
seq_length = 24
d_model = 128
vocab_size = 800

# Step-by-step memory:
input_ids: [2, 24]                    # Input token IDs
embeddings: [2, 24, 128]              # After token embedding
pos_encoded: [2, 24, 128]             # After positional encoding
transformer_out: [2, 24, 128]         # After transformer blocks
logits: [2, 24, 800]                  # After output head

# Peak memory: mainly from logits (largest tensor)
# 2 × 24 × 800 × 4 bytes = ~150KB for this batch
```

### **Key Insights:**

1. **Modular design** - Each component has a clear purpose
2. **Information flow** - Token → Embedding → Position → Transform → Predict
3. **Causal masking** - Ensures proper language modeling
4. **Parallel processing** - All positions computed simultaneously
5. **Scalable architecture** - Can adjust size by changing config parameters

### **The Complete Picture:**
Your GPT model can now:
- Convert text to numbers (tokenization)
- Understand word meanings (embeddings)
- Track word positions (positional encoding)
- Focus on relevant context (attention)
- Process information (feed forward)
- Generate predictions (output head)
- Learn from data (training loop)

Ready for **Training** - where we teach the model to predict the next word?

---

## Step 10: Training - Teaching the Model to Predict the Next Word

This is where the magic happens! We feed the model thousands of examples and it learns to understand language patterns.

```python
class GPTTrainer:
    def __init__(self, model, train_loader, val_loader, tokenizer, device):
        self.model = model
        self.train_loader = train_loader      # Training examples
        self.val_loader = val_loader          # Validation examples
        self.tokenizer = tokenizer
        self.device = device
        
        # Track progress
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
```

### **Step 10A: The Training Concept**

**Core idea:** Show the model millions of examples where it guesses the next word, then tell it if it was right or wrong.

```python
# Training example:
input_sequence = "alice was beginning to"
target_sequence = "was beginning to get"

# Model learns:
# Given "alice" → predict "was"
# Given "alice was" → predict "beginning"  
# Given "alice was beginning" → predict "to"
# Given "alice was beginning to" → predict "get"
```

### **Step 10B: Single Training Step**

```python
def train_step(self, batch):
    input_ids, targets = batch
    
    # 1. Move data to GPU/device
    input_ids = input_ids.to(self.device)  # [batch_size, seq_length]
    targets = targets.to(self.device)      # [batch_size, seq_length]
    
    # 2. Zero out previous gradients
    self.optimizer.zero_grad()
    
    # 3. Forward pass - make predictions
    logits, loss, _ = self.model(input_ids, targets)
    # logits: [batch_size, seq_length, vocab_size] - model's guesses
    # loss: scalar - how wrong the model was
    
    # 4. Backward pass - calculate gradients
    loss.backward()
    
    # 5. Clip gradients (prevent exploding)
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    
    # 6. Update weights
    self.optimizer.step()
    
    return loss.item()
```

### **Step 10C: Detailed Forward Pass During Training**

```python
# Example batch:
input_ids = [
    [8, 9, 234, 4, 67],      # "alice was beginning to get"
    [156, 23, 89, 12, 45]    # "she said hello very loud"
]
targets = [
    [9, 234, 4, 67, 12],     # "was beginning to get very"  
    [23, 89, 12, 45, 78]     # "said hello very loud again"
]

# Forward pass:
logits, loss, _ = model(input_ids, targets)

# What happens inside:
# 1. Convert IDs to embeddings
# 2. Add positional encoding
# 3. Pass through transformer layers
# 4. Get predictions for each position
# 5. Compare predictions with targets using cross-entropy loss
```

### **Step 10D: Loss Calculation Deep Dive**

```python
# For each position in each example:

# Example 1, Position 0:
input_context = "alice"
model_prediction = [0.1, 0.8, 0.05, 0.02, ...]  # Probabilities for each word
correct_answer = 9  # "was"
position_loss = -log(model_prediction[9]) = -log(0.8) = 0.22

# Example 1, Position 1:  
input_context = "alice was"
model_prediction = [0.05, 0.1, 0.7, 0.1, ...]
correct_answer = 234  # "beginning"
position_loss = -log(model_prediction[234]) = -log(0.7) = 0.36

# Total loss = average of all position losses across all examples in batch
```

### **Step 10E: Gradient Calculation and Backpropagation**

```python
# After loss.backward(), each parameter gets a gradient:

# Example: One weight in attention layer
weight_value = 0.5
gradient = -0.02  # Tells us: "decrease this weight slightly"

# Weight update:
learning_rate = 0.001
new_weight = weight_value - learning_rate * gradient
new_weight = 0.5 - 0.001 * (-0.02) = 0.5 + 0.00002 = 0.50002

# This happens for ALL 1.3 million parameters simultaneously!
```

### **Step 10F: Why Gradient Clipping Is Essential**

```python
# Without gradient clipping (BAD):
gradients = [100.0, -80.0, 150.0, -200.0, ...]  # Huge gradients!
learning_rate = 0.001
weight_updates = learning_rate * gradients = [0.1, -0.08, 0.15, -0.2, ...]
# Weights change dramatically → model becomes unstable

# With gradient clipping (GOOD):
original_gradients = [100.0, -80.0, 150.0, -200.0, ...]
gradient_norm = sqrt(100² + 80² + 150² + 200²) = 283.7
clip_norm = 1.0

if gradient_norm > clip_norm:
    clipped_gradients = gradients * (clip_norm / gradient_norm)
    clipped_gradients = [0.35, -0.28, 0.53, -0.71, ...]  # Much smaller!

# Result: Stable training
```

### **Step 10G: Complete Training Epoch**

```python
def train_epoch(self, optimizer, epoch):
    self.model.train()  # Set model to training mode
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (input_ids, targets) in enumerate(self.train_loader):
        # Move to device
        input_ids = input_ids.to(self.device)  # [8, 24] - 8 examples, 24 tokens each
        targets = targets.to(self.device)      # [8, 24]
        
        # Training step
        batch_loss = self.train_step((input_ids, targets))
        
        total_loss += batch_loss
        num_batches += 1
        
        # Print progress every 50 batches
        if batch_idx % 50 == 0:
            avg_loss = total_loss / num_batches
            print(f"Batch {batch_idx}: Loss = {batch_loss:.4f}, Avg = {avg_loss:.4f}")
    
    return total_loss / num_batches  # Average loss for the epoch
```

### **Step 10H: Validation - Testing Without Learning**

```python
def validate(self):
    self.model.eval()  # Set to evaluation mode (disables dropout)
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():  # Don't calculate gradients (saves memory)
        for input_ids, targets in self.val_loader:
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass only (no backward pass)
            logits, loss, _ = self.model(input_ids, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches
```

**Why validation matters:**
```python
# Training loss: "How well does the model do on data it has seen?"
# Validation loss: "How well does the model generalize to new data?"

# Good scenario:
train_loss = 2.1
val_loss = 2.3
# Model is learning and generalizing well

# Overfitting scenario:
train_loss = 1.2  # Very low
val_loss = 3.8    # Much higher
# Model memorized training data but can't generalize
```

### **Step 10I: Learning Rate Scheduling**

```python
# Learning rate scheduler automatically adjusts learning rate:
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

# How it works:
# Epoch 1: val_loss = 3.5, lr = 0.001
# Epoch 2: val_loss = 3.2, lr = 0.001  (improving, keep same lr)
# Epoch 3: val_loss = 3.1, lr = 0.001  (still improving)
# Epoch 4: val_loss = 3.2, lr = 0.001  (got worse, patience = 1)
# Epoch 5: val_loss = 3.3, lr = 0.001  (got worse again, patience = 0)
# Epoch 6: val_loss = 3.4, lr = 0.0005 (reduce lr by factor of 0.5)
```

### **Step 10J: Complete Training Loop**

```python
def train(self, num_epochs=5, learning_rate=1e-3):
    # Create optimizer
    optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        train_loss = self.train_epoch(optimizer, epoch)
        
        # Validation phase  
        val_loss = self.validate()
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(self.model.state_dict(), 'best_model.pth')
            print(f"New best model saved! Val Loss: {val_loss:.4f}")
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        
        # Test generation every epoch
        sample_text = self.generate_sample("alice was", max_length=10)
        print(f"Sample: '{sample_text}'")
        print("-" * 50)
```

### **Step 10K: What the Model Learns Over Time**

```python
# Epoch 1 (Random initialization):
# Input: "alice was"
# Output: "chocolate banana computer elephant"
# Loss: 4.7 (very bad)

# Epoch 2 (Starting to learn):
# Input: "alice was" 
# Output: "alice was was the the"
# Loss: 3.8 (still bad, but learning word repetition)

# Epoch 3 (Learning grammar):
# Input: "alice was"
# Output: "alice was very tired and"
# Loss: 2.1 (much better! Learning proper grammar)

# Final model:
# Input: "alice was"
# Output: "alice was beginning to get very tired of sitting"
# Loss: 1.8 (good! Generating coherent text)
```

### **Step 10L: Memory Usage During Training**

```python
# Training memory breakdown (batch_size=8, seq_length=24):

# Forward pass:
activations_memory = batch_size * seq_length * d_model * num_layers * 4_bytes
activations_memory = 8 * 24 * 128 * 4 * 4 = ~400KB

# Gradients (same size as parameters):
gradient_memory = num_parameters * 4_bytes = 1.3M * 4 = ~5MB

# Optimizer state (AdamW keeps momentum and variance for each parameter):
optimizer_memory = num_parameters * 8_bytes = 1.3M * 8 = ~10MB

# Total training memory: ~15-20MB (very manageable!)
```

### **Step 10M: Training Progress Visualization**

```python
# What you see during training:

# Epoch 1/3
# Batch 0: Loss = 4.712, Avg = 4.712
# Batch 50: Loss = 3.891, Avg = 4.201
# Batch 100: Loss = 3.456, Avg = 3.876
# Train Loss: 3.654
# Val Loss: 3.812
# Sample: 'alice was the the cat cat'

# Epoch 2/3  
# Batch 0: Loss = 3.234, Avg = 3.234
# Batch 50: Loss = 2.891, Avg = 3.045
# Batch 100: Loss = 2.567, Avg = 2.789
# Train Loss: 2.634
# Val Loss: 2.945
# Sample: 'alice was very tired of sitting'

# Epoch 3/3
# Batch 0: Loss = 2.345, Avg = 2.345
# Batch 50: Loss = 2.123, Avg = 2.234
# Batch 100: Loss = 1.987, Avg = 2.089
# Train Loss: 1.967
# Val Loss: 2.123
# Sample: 'alice was beginning to get very tired'
```

### **Key Insights:**

1. **Iterative learning** - Model gets better with each epoch
2. **Gradient-based optimization** - Small weight updates accumulate into intelligence
3. **Validation prevents overfitting** - Ensures model generalizes
4. **Loss decreases over time** - Quantitative measure of improvement
5. **Text quality improves** - Qualitative measure of learning

### **The Training Miracle:**
Through millions of tiny weight adjustments, your model learns to:
- Understand grammar rules
- Form coherent sentences  
- Follow narrative patterns
- Generate contextually appropriate text

Ready for **Text Generation** - where we see the trained model in action?

---

## Step 11: Text Generation - Seeing Your Trained Model in Action

This is the exciting part! Your trained model can now generate human-like text by predicting one word at a time.

```python
def generate_text(model, tokenizer, prompt="alice", max_length=50, temperature=1.0, top_k=10):
    model.eval()  # Set to evaluation mode
    
    # Start with the prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    generated_ids = input_ids.clone()
    
    with torch.no_grad():  # Don't need gradients for generation
        for step in range(max_length):
            # Get model predictions
            logits, _, _ = model(generated_ids)
            
            # Get predictions for the last position only
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, len(next_token_logits)))
                filtered_logits = torch.full_like(next_token_logits, float('-inf'))
                filtered_logits[top_k_indices] = top_k_logits
                next_token_logits = filtered_logits
            
            # Convert to probabilities and sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to sequence
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
    
    # Convert back to text
    return tokenizer.decode(generated_ids[0].tolist())
```

### **Step 11A: The Generation Process Step-by-Step**

```python
# Starting prompt: "alice was"
prompt = "alice was"
input_ids = [8, 9]  # "alice" = 8, "was" = 9

# Step 1: First prediction
current_sequence = [8, 9]  # "alice was"
logits, _, _ = model(current_sequence)
next_word_logits = logits[0, -1, :]  # Predictions after "was"

# Model outputs probabilities for each word:
probabilities = {
    234: 0.25,  # "beginning" - 25% probability
    67: 0.20,   # "very" - 20% probability  
    45: 0.15,   # "tired" - 15% probability
    156: 0.12,  # "sitting" - 12% probability
    89: 0.10,   # "walking" - 10% probability
    # ... other words get remaining 18%
}

# Sample: Let's say we pick "beginning" (ID 234)
current_sequence = [8, 9, 234]  # "alice was beginning"

# Step 2: Second prediction
logits, _, _ = model(current_sequence)
next_word_logits = logits[0, -1, :]  # Predictions after "beginning"

probabilities = {
    4: 0.35,    # "to" - 35% probability (very likely after "beginning")
    67: 0.20,   # "very" - 20% probability
    12: 0.15,   # "her" - 15% probability
    # ... etc
}

# Sample: Pick "to" (ID 4)
current_sequence = [8, 9, 234, 4]  # "alice was beginning to"

# Continue this process for max_length iterations...
```

### **Step 11B: Temperature - Controlling Creativity**

Temperature controls how "creative" vs "conservative" the model is:

```python
# Original logits (raw model outputs):
raw_logits = [3.0, 2.5, 1.0, 0.5, 0.2]

# Temperature = 0.1 (VERY CONSERVATIVE):
scaled_logits = [30.0, 25.0, 10.0, 5.0, 2.0]  # Divide by 0.1
probabilities = [0.91, 0.08, 0.01, 0.00, 0.00]  # Almost always picks best word
# Output: "alice was beginning to get very tired of sitting by her sister"

# Temperature = 1.0 (BALANCED):
scaled_logits = [3.0, 2.5, 1.0, 0.5, 0.2]  # No change
probabilities = [0.50, 0.31, 0.11, 0.07, 0.03]  # Reasonable distribution
# Output: "alice was beginning to feel quite sleepy and drowsy"

# Temperature = 2.0 (VERY CREATIVE):
scaled_logits = [1.5, 1.25, 0.5, 0.25, 0.1]  # Divide by 2.0
probabilities = [0.31, 0.26, 0.17, 0.14, 0.12]  # Much more random
# Output: "alice was purple elephants dancing rainbow yesterday mountains"
```

### **Step 11C: Top-K Sampling - Quality Control**

Top-K sampling only considers the K most likely words:

```python
# All word probabilities:
all_probs = {
    234: 0.25,  # "beginning"
    67: 0.20,   # "very"  
    45: 0.15,   # "tired"
    156: 0.12,  # "sitting"
    89: 0.10,   # "walking"
    23: 0.05,   # "happy"
    78: 0.03,   # "sad"
    445: 0.02,  # "elephant"  ← Weird word!
    167: 0.01,  # "purple"    ← Very weird!
    # ... 791 more words with tiny probabilities
}

# Top-K = 5 sampling:
# Only consider top 5 words: [234, 67, 45, 156, 89]
# Renormalize their probabilities:
top_k_probs = {
    234: 0.25/0.82 = 0.30,  # "beginning"
    67: 0.20/0.82 = 0.24,   # "very"
    45: 0.15/0.82 = 0.18,   # "tired"  
    156: 0.12/0.82 = 0.15,  # "sitting"
    89: 0.10/0.82 = 0.12,   # "walking"
}

# Benefits:
# - Prevents weird words like "elephant" or "purple"
# - Maintains creativity within reasonable bounds
# - Much better text quality
```

### **Step 11D: Different Generation Strategies**

```python
# 1. GREEDY DECODING (always pick best word):
def greedy_decode():
    probs = F.softmax(logits, dim=-1)
    next_token = torch.argmax(probs)  # Always pick highest probability
    # Result: Deterministic but boring
    # "alice was beginning to get very tired of sitting by her sister on the bank"

# 2. RANDOM SAMPLING:
def random_sample():
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)  # Random according to probabilities
    # Result: Creative but sometimes incoherent
    # "alice was beginning to purple elephant dance mountain yesterday"

# 3. TOP-K + TEMPERATURE (our approach):
def top_k_temperature_sample():
    # Apply temperature
    scaled_logits = logits / temperature
    # Apply top-k filtering
    top_k_logits, top_k_indices = torch.topk(scaled_logits, k)
    # Sample from filtered distribution
    # Result: Good balance of quality and creativity
    # "alice was beginning to feel quite drowsy and sleepy in the warm afternoon sun"
```

### **Step 11E: Real Example - Before vs After Training**

```python
# BEFORE TRAINING (random weights):
prompt = "alice was"
# Model output: "alice was purple mountain chocolate elephant computer banana"
# Explanation: Model has no understanding, just random word generation

# AFTER TRAINING (learned patterns):
prompt = "alice was"  
# Model output: "alice was beginning to get very tired of sitting by her sister"
# Explanation: Model learned:
# - "alice" is often followed by "was" (subject-verb pattern)
# - "beginning to" is a common phrase
# - "tired of sitting" makes semantic sense
# - Overall sentence structure follows English grammar
```

### **Step 11F: What the Model Actually Learned**

Through training, your model internalized these patterns:

```python
# 1. GRAMMATICAL PATTERNS:
# Subject → Verb: "alice" → "was", "cat" → "sat"
# Article → Noun: "the" → "cat", "a" → "book"  
# Adjective → Noun: "big" → "house", "red" → "car"

# 2. SEMANTIC RELATIONSHIPS:
# Actions → Objects: "sat" → "chair/mat", "read" → "book"
# Spatial: "on" → "table/floor", "in" → "house/box"
# Temporal: "then" → past_events, "will" → future_events

# 3. NARRATIVE FLOW:
# Story beginnings: "once upon" → "a time"
# Character actions: "alice" → walking/sitting/thinking
# Dialogue patterns: "said" → quotes, questions → answers

# 4. WORLD KNOWLEDGE:
# People sit on chairs, not walls
# Books are read, not eaten  
# Day comes before night
# Characters have consistent behaviors
```

### **Step 11G: Generation Quality Metrics**

```python
# How to evaluate generation quality:

# 1. PERPLEXITY (mathematical measure):
# Lower perplexity = better predictions
# Before training: perplexity ≈ 800 (terrible)
# After training: perplexity ≈ 45 (good)

# 2. HUMAN EVALUATION:
# Fluency: Does it sound natural? (1-5 scale)
# Coherence: Does it make sense? (1-5 scale)  
# Relevance: Does it follow from the prompt? (1-5 scale)

# 3. AUTOMATED METRICS:
# BLEU score: Compared to reference text
# ROUGE score: Content overlap measures
# Sentence similarity: Semantic coherence
```

### **Step 11H: Interactive Generation Example**

```python
# Live generation session:

print("GPT Model Ready! Type prompts to generate text.")

while True:
    prompt = input("Enter prompt: ")
    if prompt == "quit":
        break
    
    # Generate with different settings
    conservative = generate_text(model, tokenizer, prompt, max_length=20, temperature=0.7, top_k=5)
    creative = generate_text(model, tokenizer, prompt, max_length=20, temperature=1.2, top_k=15)
    
    print(f"Conservative: {conservative}")
    print(f"Creative: {creative}")
    print("-" * 50)

# Example session:
# Enter prompt: alice was
# Conservative: alice was beginning to get very tired of sitting by her sister on the bank
# Creative: alice was feeling quite drowsy and started to wonder about the peculiar rabbit

# Enter prompt: once upon
# Conservative: once upon a time there was a little girl who lived in a small house
# Creative: once upon a magical evening, strange creatures began dancing under the moonlight
```

### **Step 11I: Memory Usage During Generation**

```python
# Generation is much lighter than training:

# No gradients needed: 0 MB (vs ~10MB during training)
# No optimizer state: 0 MB (vs ~10MB during training)  
# Only forward pass: ~1MB for activations
# Growing sequence: starts small, grows with each token

# Peak memory for 50-token generation: ~2-3MB total
# Very efficient! Can run on phone/laptop easily
```

### **Step 11J: Generation Speed**

```python
# Generation speed depends on model size:

# Our small model (1.3M parameters):
# ~100-200 tokens/second on CPU
# ~500-1000 tokens/second on GPU

# For comparison:
# GPT-3 (175B parameters): ~20-50 tokens/second
# Our model is much faster because it's much smaller!

# Generation time for 50 tokens:
# CPU: ~0.3 seconds
# GPU: ~0.05 seconds
```

### **Step 11K: Common Generation Issues and Solutions**

```python
# PROBLEM 1: Repetition
# Output: "alice was was was was was"
# Solution: Add repetition penalty or use different sampling

# PROBLEM 2: Incoherence  
# Output: "alice was purple elephant mountain"
# Solution: Lower temperature, use top-k sampling

# PROBLEM 3: Too boring
# Output: "alice was tired alice was tired alice was tired"
# Solution: Increase temperature, increase top-k

# PROBLEM 4: Doesn't follow prompt
# Output: Prompt="alice was happy" → "bob went shopping"
# Solution: Better training data, longer context
```

### **Key Insights:**

1. **Autoregressive generation** - One word at a time, each depends on previous
2. **Probabilistic sampling** - Model outputs probabilities, we sample from them
3. **Quality vs creativity tradeoff** - Temperature and top-k control this balance
4. **Learned patterns emerge** - Training creates understanding of language structure
5. **Interactive capability** - Model can respond to any prompt in real-time

### **The Generation Miracle:**
Your 1.3M parameter model can now:
- Complete any sentence you start
- Write coherent stories
- Follow grammatical rules
- Maintain narrative consistency
- Generate creative but sensible text

**Final result:** You've built a miniature GPT that understands language and can generate human-like text!

This is the same fundamental technology behind ChatGPT, GPT-4, and other large language models - just scaled up with more parameters, more data, and more compute!

---

