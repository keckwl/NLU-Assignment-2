import torch
import torch.nn as nn
import torch.nn.functional as F


def _sample_with_penalty(logits, temperature, generated, penalty=5.0):
    # Temperature sampling with repetition penalty on the last 4 generated tokens
    logits = logits.clone()
    for tok_id in set(generated[-4:]):
        logits[0, tok_id] -= penalty
    return torch.multinomial(F.softmax(logits / temperature, dim=-1), 1)


class VanillaRNN(nn.Module):
    # Single-layer RNN: Embedding -> RNN -> Linear
    def __init__(self, vocab_size, embed_dim=32, hidden_size=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.embedding   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        out, hidden = self.rnn(self.embedding(x), hidden)
        return self.fc(out), hidden

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, sos_idx, eos_idx, max_len=20, temperature=1.0, device="cpu"):
        self.eval()
        tok, hidden, generated = torch.tensor([[sos_idx]], dtype=torch.long, device=device), None, []
        for _ in range(max_len):
            logits, hidden = self.forward(tok, hidden)
            nxt = _sample_with_penalty(logits[:, -1, :], temperature, generated)
            if nxt.item() == eos_idx:
                break
            generated.append(nxt.item())
            tok = nxt
        return generated


class BidirectionalLSTM(nn.Module):
    # BiLSTM encoder projects fwd+bwd hidden to init a unidirectional LSTM decoder
    def __init__(self, vocab_size, embed_dim=32, hidden_size=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.embedding   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True, bidirectional=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        # Project concatenated fwd+bwd hidden/cell to single hidden_size
        self.hidden_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.cell_proj   = nn.Linear(hidden_size * 2, hidden_size)
        self.decoder = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def encode(self, x):
        _, (h, c) = self.encoder(self.embedding(x))
        batch = x.size(0)
        h = h.view(self.num_layers, 2, batch, self.hidden_size)
        c = c.view(self.num_layers, 2, batch, self.hidden_size)
        h = torch.tanh(self.hidden_proj(torch.cat([h[:, 0], h[:, 1]], dim=-1)))
        c = torch.tanh(self.cell_proj( torch.cat([c[:, 0], c[:, 1]], dim=-1)))
        return h, c

    def forward(self, x, hidden=None):
        h, c = self.encode(x)
        out, (h, c) = self.decoder(self.embedding(x), (h, c))
        return self.fc(out), (h, c)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, sos_idx, eos_idx, max_len=20, temperature=1.0, device="cpu"):
        self.eval()
        seed = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
        h, c = self.encode(seed)
        tok, generated = seed, []
        for _ in range(max_len):
            out, (h, c) = self.decoder(self.embedding(tok), (h, c))
            nxt = _sample_with_penalty(self.fc(out[:, -1, :]), temperature, generated)
            if nxt.item() == eos_idx:
                break
            generated.append(nxt.item())
            tok = nxt
        return generated


class Attention(nn.Module):
    # Additive attention: context = weighted sum of encoder outputs
    def __init__(self, hidden_size):
        super().__init__()
        self.W_enc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_dec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v     = nn.Linear(hidden_size, 1,           bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        dec_exp = decoder_hidden.unsqueeze(1).expand_as(encoder_outputs)
        energy  = torch.tanh(self.W_enc(encoder_outputs) + self.W_dec(dec_exp))
        weights = F.softmax(self.v(energy).squeeze(-1), dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class RNNWithAttention(nn.Module):
    # RNN encoder-decoder with additive attention; decoder input = [embedding || context]
    def __init__(self, vocab_size, embed_dim=32, hidden_size=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.embedding   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.RNN(input_size=embed_dim, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0)
        self.attention = Attention(hidden_size)
        # Aliases for vectorized forward pass
        self.W_enc = self.attention.W_enc
        self.W_dec = self.attention.W_dec
        self.v     = self.attention.v
        self.decoder = nn.RNN(input_size=embed_dim + hidden_size, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # Vectorized attention over all time steps at once
        emb = self.embedding(x)
        enc_out, enc_hidden = self.encoder(emb, hidden)
        B, T, H = enc_out.shape
        query_exp = enc_hidden[-1].unsqueeze(1).expand(B, T, H)
        energy    = torch.tanh(self.W_enc(enc_out) + self.W_dec(query_exp))
        weights   = F.softmax(self.v(energy).squeeze(-1), dim=-1)
        context   = torch.bmm(weights.unsqueeze(1), enc_out).expand(B, T, H)
        dec_out, dec_hidden = self.decoder(torch.cat([emb, context], dim=-1), enc_hidden)
        return self.fc(dec_out), dec_hidden

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, sos_idx, eos_idx, max_len=20, temperature=1.0, device="cpu"):
        self.eval()
        seed = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
        enc_out, dec_hidden = self.encoder(self.embedding(seed))
        tok, generated = seed, []
        for _ in range(max_len):
            emb_t      = self.embedding(tok)
            context, _ = self.attention(enc_out, dec_hidden[-1])
            dec_out, dec_hidden = self.decoder(torch.cat([emb_t, context.unsqueeze(1)], dim=-1), dec_hidden)
            nxt = _sample_with_penalty(self.fc(dec_out[:, -1, :]), temperature, generated)
            if nxt.item() == eos_idx:
                break
            generated.append(nxt.item())
            tok = nxt
        return generated
