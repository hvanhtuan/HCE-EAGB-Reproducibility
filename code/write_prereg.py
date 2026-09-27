"""Ghi tệp tiền đăng ký trước khi mở tập kiểm định. Chạy SAU thí điểm KB2, TRƯỚC 'kb2_h1.py test'."""
import json, hashlib, datetime
pil = json.load(open("../results/kb2_pilot.json"))
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
H1 = {
    "locked_at_utc": now,
    "data_split": {"train": [2012, 2018], "pilot": [2019, 2020], "test": [2021, 2023], "shift": [2024]},
    "xi_eval": 1.5,
    "k_selected": pil["meta"]["k_selected"], "xi_selected": pil["meta"]["xi_selected"], "rounds": pil["meta"]["rounds"], "level_factor": pil["meta"]["level_factor"],
    "secondary_analysis": "cân chỉnh mức toàn cục: nhân dự báo của mỗi mô hình với hệ số O/E của mô hình đó trên 2019–2020 (hệ số khóa ở đây); báo cáo cùng tiêu chí H1a–H1c, không dùng cho quyết định chính",
    "primary_comparison": "EAGB (HCE đồng thời) vs GLM Tweedie (đối chứng chính)",
    "H1a": {"stat": "(D_EAGB - D_GLM)/D_GLM, bootstrap cụm quận B=2000", "supported_if": "cận trên một phía 95% < -0.01 (MEI 1%)"},
    "H1b": {"rule": "đồng thời: |OE_E-1| <= |OE_G-1| + 0.05; |slope_E-1| <= |slope_G-1| + 0.10; maxdec_E <= maxdec_G + 0.10 (ước lượng điểm trên tập kiểm định)"},
    "H1c": {"stat": "(D_EAGB - D_flat)/D_flat", "supported_if": "cận trên một phía 95% < -0.005 (MEI 0,5%)"},
    "GT2": {"stat": "(D_EAGB - D_seq)/D_seq", "supported_if": "cận trên một phía 95% < +0.005 (không thua kém)", "refuted_if": "cận dưới một phía 95% > +0.005",
            "otherwise": "chưa đủ bằng chứng"},
    "H1_overall": "được ủng hộ khi H1a và H1b và H1c cùng đạt; kiểm định theo thứ bậc H1a -> H1b -> H1c -> GT2",
    "H2": {
        "H2a": {"stat": "tỷ số độ phân tán chuẩn hóa Owen/Shapley-tổng-nhóm dưới nhiễu mô hình (8 bootstrap cụm quận, 100 điểm)",
                "supported_if": "cận trên một phía 95% (bootstrap theo điểm) < 0.95", "refuted_if": "cận dưới hai phía 95% > 1.05"},
        "H2a_secondary": "cùng tiêu chí cho nhiễu nền (mô tả, không dùng cho quyết định)",
        "H2b": {"stat": "AUC_Owen - AUC_SHAP trong KB4c (10 lần lặp)", "supported_if": "cận dưới một phía 95% > -0.05 (không thua kém)"},
        "H2c": "đồng thuận chuyên gia — chưa thực hiện",
        "fidelity_NI": "R2 trung thành trung bình của Owen không thấp hơn Shapley quá 0.05",
        "H2_overall": "được ủng hộ (một phần, không gồm H2c) khi H2a và H2b cùng đạt"},
    "H3": {
        "stat": "tỷ số phương sai có trọng số của Δlog Π (bảo thủ/danh nghĩa), gộp 2021–2023, trung bình 5 hạt giống; bootstrap cụm quận B=1000",
        "supported_if": "cận trên một phía 95% < 0.95", "refuted_if": "cận dưới hai phía 95% > 1.05",
        "TOST": "lợi nhuận kỳ vọng: cận dưới một phía 95% (t, theo hạt giống) của (cons-nom)/|nom| > -0.02; tái tục: cận dưới của (cons-nom) > -0.005",
        "H3_overall": "được ủng hộ khi tỷ số phương sai đạt và cả hai điều kiện TOST đạt"},
    "RQ0": "ít nhất một trong (H1a, H2a, H3-tỷ số phương sai) được ủng hộ VÀ không có trục nào vi phạm điều kiện không thua kém (H1b; fidelity_NI; TOST)",
}
H3 = {"locked_at_utc": now, "Npop": 48, "Tgen": 120, "seeds": [11, 22, 33, 44, 55], "alpha_conformal": 0.10, "gamma_main": 2.0,
      "gamma_kb8": [0.5, 1.0, 4.0], "selection_rule": "khoảng cách Euclid nhỏ nhất tới điểm lý tưởng sau chuẩn hóa min–max của mặt trội",
      "delta_rule": "bội nhỏ nhất của 0,05 không nhỏ hơn 1,05 × nửa độ rộng lớn nhất của log(Π0/ŝ) trong một ô biểu phí ở năm thí điểm 2020",
      "eta_rule": "giá trị nhỏ nhất trong {0,70;0,75;0,80;0,85;0,90} mà β0 khả thi cho cả bài toán danh nghĩa và bảo thủ ở năm 2020",
      "box": [0.0, 1.5], "load0": "log(1/0.65)"}
for nm, obj in [("prereg_H1.json", H1), ("prereg_H3_rules.json", H3)]:
    s = json.dumps(obj, indent=1, ensure_ascii=False)
    open("../results/" + nm, "w").write(s)
    print(nm, hashlib.sha256(s.encode()).hexdigest(), now)
