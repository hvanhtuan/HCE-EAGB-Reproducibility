"""Tái cài đặt ĐỘC LẬP của kiến trúc "Credibility Transformer" (Richman, Scognamiglio & Wüthrich,
"Credibility Transformer: A Novel Approach for Tabular Data Prediction", European Actuarial
Journal, 2025; bản arXiv: https://arxiv.org/abs/2409.16653), dùng làm đối chứng học sâu hiện đại
cho KB1 (freMTPL2).

GHI CHÚ BẢN QUYỀN — QUAN TRỌNG: mã nguồn gốc của tác giả (github.com/RonRichman/CredibilityTransformer)
được công bố dưới giấy phép Creative Commons BY-NC-ND 4.0 (NoDerivatives): giấy phép đó KHÔNG cho
phép tạo bản chỉnh sửa/thích ứng (adapted material) từ mã nguồn của họ. Mô-đun này vì vậy KHÔNG
sao chép hay chỉnh sửa mã nguồn của tác giả; nó được viết lại từ đầu, chỉ dựa trên mô tả kiến
trúc bằng công thức/toán học trong chính bài báo (Mục 2-3), theo đúng cấu hình "cơ bản" mà bài
báo dùng cho thí nghiệm freMTPL2 (Mục 3):
  - b = 5 (chiều embedding mỗi token trước vị trí); nối với embedding vị trí (chiều b) -> 2b = 10
  - 1 tầng Transformer encoder, 1 đầu chú ý (multi-head attention với M=1)
  - FNN sau chú ý: ẩn 32 nơ-ron, kích hoạt GELU, chiếu lại 2b=10
  - Dropout 1% (chỉ khi huấn luyện)
  - Token CLS học được, hai nhánh: "trước" (prior, không qua chú ý — đại diện trung bình danh mục)
    và "sau" (trans, đã qua Transformer — mang thông tin hiệp biến)
  - Trộn tín nhiệm khi huấn luyện: mỗi bước lấy mẫu Z~Bernoulli(alpha) độc lập,
    c_cred = Z * c_trans + (1-Z) * c_prior; khi dự báo cố định Z=1 (chỉ dùng c_trans)
  - Bộ giải mã nông: Linear(2b, 16) -> GELU -> Linear(16, 1), đầu ra ở thang log tần suất
  - Hàm mất mát: độ lệch Poisson có trọng số phơi nhiễm (công thức 3.1 của bài báo)
  - alpha = 0,90; kích thước lô 1024; bộ tối ưu NAdam; dropout 1%; tập hợp (ensemble) nhiều lần
    chạy độc lập lấy trung bình dự báo

Vì đây là một TÁI CÀI ĐẶT (không phải mã của tác giả), một số chi tiết không được nêu tường minh
trong văn bản bài báo (ví dụ dạng chính xác của "prior token" khi không có tầng chú ý, chi tiết
chuẩn hóa) được chọn theo cách hợp lý nhất, gần nhất với mô tả — và được ghi chú rõ trong code.
Kết quả vì vậy là một đối chứng "theo tinh thần" kiến trúc gốc, không phải một tái lập số liệu
byte-for-byte của bài báo gốc; đây là hạn chế trung thực cần nêu khi báo cáo.
"""
import numpy as np
import torch
import torch.nn as nn


class ContinuousEmbedding(nn.Module):
    """z^(1): R -> R^b (tuyến tính), z^(2): R^b -> R^b (tanh) — Mục 2.1 của bài báo."""

    def __init__(self, b):
        super().__init__()
        self.l1 = nn.Linear(1, b)
        self.l2 = nn.Linear(b, b)

    def forward(self, x):  # x: (batch,)
        h = self.l1(x.unsqueeze(-1))
        return torch.tanh(self.l2(h))


class CredibilityTransformer(nn.Module):
    def __init__(self, cat_cardinalities, n_continuous, b=5, dropout=0.01, ff_hidden=32, dec_hidden=16):
        super().__init__()
        self.b = b
        self.cat_embeds = nn.ModuleList([nn.Embedding(card, b) for card in cat_cardinalities])
        self.cont_embeds = nn.ModuleList([ContinuousEmbedding(b) for _ in range(n_continuous)])
        T = len(cat_cardinalities) + n_continuous
        self.T = T
        self.pos_embed = nn.Embedding(T, b)
        self.cls = nn.Parameter(torch.zeros(1, 1, 2 * b))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=2 * b, nhead=1, dim_feedforward=ff_hidden,
                                           dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.decoder = nn.Sequential(nn.Linear(2 * b, dec_hidden), nn.GELU(), nn.Linear(dec_hidden, 1))

    def embed_sequence(self, x_cat, x_cont):
        toks = [emb(x_cat[:, j]) for j, emb in enumerate(self.cat_embeds)]
        toks += [emb(x_cont[:, j]) for j, emb in enumerate(self.cont_embeds)]
        x = torch.stack(toks, dim=1)  # (batch, T, b)
        pos = self.pos_embed(torch.arange(self.T, device=x.device)).unsqueeze(0).expand(x.size(0), -1, -1)
        return torch.cat([x, pos], dim=-1)  # (batch, T, 2b) — nối (concat), không cộng

    def forward(self, x_cat, x_cont, alpha, training_mix=True):
        seq = self.embed_sequence(x_cat, x_cont)
        batch = seq.size(0)
        c_prior = self.cls.expand(batch, -1, -1).squeeze(1)  # CLS thô, KHÔNG qua chú ý (Mục 2.2, "prior")
        seq_plus = torch.cat([seq, self.cls.expand(batch, -1, -1)], dim=1)  # nối CLS vào cuối chuỗi
        out = self.encoder(seq_plus)
        c_trans = out[:, -1, :]  # CLS sau Transformer (Mục 2.2, "trans")
        if training_mix and self.training:
            z = (torch.rand(batch, 1, device=seq.device) < alpha).float()
            c_cred = z * c_trans + (1 - z) * c_prior
        else:
            c_cred = c_trans  # dự báo: Z ≡ 1 (chỉ dùng thông tin đã qua chú ý)
        return self.decoder(c_cred).squeeze(-1)  # log-mu


def poisson_dev_loss(log_mu, y, v):
    """Độ lệch Poisson có trọng số phơi nhiễm (công thức 3.1 của bài báo), trung bình theo lô."""
    mu = torch.exp(log_mu)
    lam = v * mu
    # giới hạn dưới để tránh log(0); y=0 thì số hạng y*log(.) = 0 theo quy ước độ lệch Poisson
    ll = lam - y - torch.where(y > 0, y * torch.log(torch.clamp(lam, min=1e-8) / torch.clamp(y, min=1e-8)),
                               torch.zeros_like(y))
    return 2.0 * ll.mean()


def fit_predict_one(Xcat_tr, Xcont_tr, y_tr, v_tr, Xcat_va, Xcont_va, y_va, v_va, Xcat_te, Xcont_te,
                    cat_cardinalities, n_continuous, seed, b=5, alpha=0.90, batch_size=1024,
                    max_epochs=150, patience=15, dropout=0.01, lr=1e-3, device="cpu"):
    """Huấn luyện MỘT lần chạy (một hạt giống) với dừng sớm trên tập validation, trả về log-mu dự
    báo trên tập kiểm định (đã chuyển sang numpy)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = CredibilityTransformer(cat_cardinalities, n_continuous, b=b, dropout=dropout).to(device)
    opt = torch.optim.NAdam(model.parameters(), lr=lr)

    def to_t(a, dtype=torch.float32):
        return torch.as_tensor(a, dtype=dtype, device=device)

    Xcat_tr_t, Xcont_tr_t = to_t(Xcat_tr, torch.long), to_t(Xcont_tr)
    y_tr_t, v_tr_t = to_t(y_tr), to_t(v_tr)
    Xcat_va_t, Xcont_va_t = to_t(Xcat_va, torch.long), to_t(Xcont_va)
    y_va_t, v_va_t = to_t(y_va), to_t(v_va)
    n = len(y_tr)
    best_val, best_state, bad = np.inf, None, 0
    rng = np.random.default_rng(seed)
    for epoch in range(max_epochs):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            log_mu = model(Xcat_tr_t[idx], Xcont_tr_t[idx], alpha, training_mix=True)
            loss = poisson_dev_loss(log_mu, y_tr_t[idx], v_tr_t[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_log_mu = model(Xcat_va_t, Xcont_va_t, alpha, training_mix=False)
            val_loss = poisson_dev_loss(val_log_mu, y_va_t, v_va_t).item()
        if val_loss < best_val - 1e-5:
            best_val, best_state, bad = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        Xcat_te_t, Xcont_te_t = to_t(Xcat_te, torch.long), to_t(Xcont_te)
        te_log_mu = model(Xcat_te_t, Xcont_te_t, alpha, training_mix=False).cpu().numpy()
    n_params = sum(p.numel() for p in model.parameters())
    return te_log_mu, dict(best_val_dev=best_val, epochs_run=epoch + 1, n_params=n_params)
