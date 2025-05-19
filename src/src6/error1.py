input_file = "D:/master/src6/5_both_null_reset_t10_g10000_r100_w50_b1.csv"
output_file = "D:/master/src6/5_both_null_reset_t10_g10000_r100_w50_b1_fixed.csv"
target_line_number = 2495629

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for current_line_number, line in enumerate(fin, 1):
        if current_line_number == target_line_number:
            parts = line.strip().split(",")
            if len(parts) == 8:
                # 不要な先頭の "3" を削除して再構成
                line = ",".join(parts[1:]) + "\n"
        fout.write(line)