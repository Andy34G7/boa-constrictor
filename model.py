import torch
import torch.nn as nn
import numpy as np

class BoaBytePredictor_LSTM(nn.Module):
    """ LSTM model adapted to predict the next byte in a sequence. """
    def __init__(self, d_model=256, num_layers=4, vocab_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, vocab_size)
        )

    def forward(self, x, inference_params=None):
        out, _ = self.lstm(self.embedding(x))
        return self.head(out)
        
    @torch.inference_mode()
    def init_stream(self, max_len: int, batch_size: int = 1, device=None, dtype=None):
        device = device or self.embedding.weight.device
        h_0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
        c_0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
        # Return a list instead of a tuple so that codec.py can just update the logits with a single call.
        return [h_0, c_0]

    @torch.inference_mode()
    def step(self, byte_t: torch.LongTensor, caches) -> torch.Tensor:
        # LSTM expects a tuple (h, c)
        out, (new_h, new_c) = self.lstm(self.embedding(byte_t).unsqueeze(1), (caches[0], caches[1]))  
        # Mutate the caller's list in-place to carry the state forward
        caches[0] = new_h
        caches[1] = new_c
        return self.head(out.squeeze(1))

def BoaConstrictor(d_model=256, num_layers=4, vocab_size=256, device="cuda"):
    """ Construct a BoaBytePredictor with smaller model size for Boa experiments. """
    device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
    return BoaBytePredictor_LSTM(d_model, num_layers, vocab_size).to(device)

# These shouldn't require any changes (I hope)

def _aligned_len(n_bytes: int, seq_len: int, batch_size: int) -> int:
    # number of usable bytes that fit whole (batch_size * seq_len) chunks
    block = seq_len * batch_size
    return (n_bytes // block) * block

def make_splits(data_bytes: bytes | np.ndarray, seq_len: int, batch_size: int,
                splits=(0.8, 0.1, 0.1)):
    assert abs(sum(splits) - 1.0) < 1e-6, "splits must sum to 1.0"
    buf = np.frombuffer(bytes(data_bytes), dtype=np.uint8)
    usable = _aligned_len(len(buf), seq_len, batch_size)
    buf = buf[:usable]

    n = len(buf)
    n_train = _aligned_len(int(n * splits[0]), seq_len, batch_size)
    n_val   = _aligned_len(int(n * splits[1]), seq_len, batch_size)
    n_test  = _aligned_len(n - n_train - n_val, seq_len, batch_size)

    i0, i1, i2 = 0, n_train, n_train + n_val
    train_bytes = buf[i0:i1].tobytes()
    val_bytes   = buf[i1:i2].tobytes()
    test_bytes  = buf[i2:i2+n_test].tobytes()

    return train_bytes, val_bytes, test_bytes

class ByteDataloader:
    """ Simple dataloader that yields batches of bytes. """
    def __init__(self, data_bytes, seq_len=1048576, batch_size=1, device="cuda"):
        self.data_bytes = np.frombuffer(data_bytes, dtype=np.uint8)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pos = 0
        self.device = device
    def __len__(self):
        """ Returns the total number of batches in the dataset. """
        return len(self.data_bytes) // (self.seq_len * self.batch_size)
    def __iter__(self):
        return self
    def __next__(self):
        if self.pos + self.seq_len * self.batch_size > len(self.data_bytes):
            self.pos = 0  # reset for simplicity
            raise StopIteration
        
        batch_indices = np.arange(self.pos, self.pos + self.seq_len * self.batch_size)
        batch_indices = batch_indices.reshape(self.batch_size, self.seq_len)
        self.pos += self.seq_len * self.batch_size
        
        batch = self.data_bytes[batch_indices]
        return torch.tensor(batch, dtype=torch.long).to(self.device)
    
