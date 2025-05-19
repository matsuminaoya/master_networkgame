target_line_number = 2495629
csv_path = "D:/master/src6/5_both_null_reset_t10_g10000_r100_w50_b1_fixed.csv"

with open(csv_path, encoding='utf-8') as f:  # 必要に応じて encoding='shift_jis' などに変更
    for current_line_number, line in enumerate(f, 1):
        if current_line_number == target_line_number:
            print(f"{current_line_number}行目の内容:")
            print(line)
            break