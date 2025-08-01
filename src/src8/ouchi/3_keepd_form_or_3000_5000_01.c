#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mt19937ar.h"

#define NUM_AGENTS 100  // TODO:

typedef struct {
  float tc;
  float tf;
  int is_cooperator;
} Agent;

// エージェントのインデックスをシャッフルするユーティリティ
// 社会学習の処理順を変えるためだけ問題なし
// Fisher-Yatesアルゴリズム/Knuthシャッフル
void shuffle(int *array, int n) {
  for (int i = n - 1; i > 0; i--) {     // iを後ろから前に1ずつ動かす
    int j = genrand_int32() % (i + 1);  // 0からiまでの数字をランダムにj
    int temp = array[i];                // arrayのi番目を
    array[i] = array[j];                // arrayのj番目と
    array[j] = temp;                    // 入れ替える
  }
}

#define MODEL_TYPE 1  // 0: 切る優先（モデルA）、1: 張る優先（モデルB）

void part5() {
  // CSVファイルを出力するための準備 TODO:
  FILE *csv_file =
      fopen("3_keepd_01_form_or_t10_g10000_r100_w5000_b1.csv", "w");

  init_genrand((unsigned long)time(NULL));  // 乱数

  float density = 0.1f;  // [0.0, 1.0] 初期ネットワーク密度 // TODO:

  float benefit = 2.0f;    // 協力者が与える利得
  float cost = 1.0f;       // 協力にかかるコスト
  float beta = 1.0f;       // フェルミ関数の鋭さ TODO:
  float mutation = 0.01f;  // 突然変異確率

  int trial = 10;          // TODO:
  int generation = 10000;  // TODO:
  int round = 100;         // TODO:
  int work = 5000;         // TODO:

  for (int tr = 0; tr < trial; tr++) {  // トライアル開始
    printf("Trial %d:\n", tr);
    Agent agents[NUM_AGENTS];
    for (int i = 0; i < NUM_AGENTS; i++) {
      agents[i].tc = 0.0f;
      agents[i].tf = 0.0f;
      agents[i].is_cooperator = 0;
    }  // tc,tl,tf,の初期化 // 0スタート特有 TODO:

    // マトリクス初期化
    int link_matrix[NUM_AGENTS][NUM_AGENTS];
    memset(link_matrix, 0, sizeof(link_matrix));
    // 初期リンク密度によるリンク構築
    int max_possible_links = NUM_AGENTS * (NUM_AGENTS - 1) / 2;
    int target_link_count = (int)(density * max_possible_links);
    // 全てのリンク候補を生成（i < j のみ）
    int link_candidates[max_possible_links][2];
    int idx = 0;
    for (int i = 0; i < NUM_AGENTS; i++) {
      for (int j = i + 1; j < NUM_AGENTS; j++) {
        link_candidates[idx][0] = i;
        link_candidates[idx][1] = j;
        idx++;
      }
    }
    // ランダムにシャッフル
    for (int i = max_possible_links - 1; i > 0; i--) {
      int j = genrand_int32() % (i + 1);
      int temp0 = link_candidates[i][0];
      int temp1 = link_candidates[i][1];
      link_candidates[i][0] = link_candidates[j][0];
      link_candidates[i][1] = link_candidates[j][1];
      link_candidates[j][0] = temp0;
      link_candidates[j][1] = temp1;
    }
    // 最初の target_link_count 個を選んでリンクを張る
    for (int i = 0; i < target_link_count; i++) {
      int a = link_candidates[i][0];
      int b = link_candidates[i][1];
      link_matrix[a][b] = 1;
      link_matrix[b][a] = 1;  // 無向リンク
    }

    for (int ge = 0; ge < generation; ge++) {  // 世代開始
      int count_game_ge[NUM_AGENTS] = {0};
      int count_coop_game_ge[NUM_AGENTS] = {0};
      float count_poff_ge[NUM_AGENTS] = {0.0f};

      for (int ro = 0; ro < round; ro++) {  // ラウンド開始
        int link_count[NUM_AGENTS] = {0};
        for (int i = 0; i < NUM_AGENTS; i++) {
          for (int j = 0; j < NUM_AGENTS; j++) {
            if (link_matrix[i][j] == 1) {
              link_count[i]++;
            }
          }
        }  // リンク数調べ終了

        if (ro == 0) {
          for (int i = 0; i < NUM_AGENTS; i++) {
            float r_0rocd = (float)genrand_real1();  // [0,1]
            if (agents[i].tc <= r_0rocd) {
              agents[i].is_cooperator = 1;
            } else {
              agents[i].is_cooperator = 0;
            }
          }  // 協力非協力決め終了
        }  // 初回ラウンドは乱数で決める
        if (ro > 0) {
          for (int i = 0; i < NUM_AGENTS; i++) {
            if (link_count[i] > 0) {
              int coop_link_count[NUM_AGENTS] = {0};
              for (int i = 0; i < NUM_AGENTS; i++) {
                for (int j = 0; j < NUM_AGENTS; j++) {
                  if (link_matrix[i][j] == 1 && agents[j].is_cooperator == 1) {
                    coop_link_count[i]++;
                  }
                }
              }  // リンク協力者数数え終了（初回ラウンドは要らない）
              float coop_ratio = (float)coop_link_count[i] / link_count[i];
              if (agents[i].tc <= coop_ratio) {
                agents[i].is_cooperator = 1;
              } else {
                agents[i].is_cooperator = 0;
              }
            } else {  // リンクがない者は乱数と比較
              float r_nolcd = (float)genrand_real1();  // [0,1]
              if (agents[i].tc <= r_nolcd) {
                agents[i].is_cooperator = 1;
              } else {
                agents[i].is_cooperator = 0;
              }
            }
          }  // 協力非協力決め終了
        }  // 初回ラウンド以外

        float received_ro[NUM_AGENTS] = {
            0.0f};  // 各エージェントが受け取った利得
        float payoff_ro[NUM_AGENTS] = {
            0.0f};  // 各エージェントの最終利得（リンク数割り）
        // 協力者が利得を配り、自分はコストを負担
        for (int i = 0; i < NUM_AGENTS; i++) {
          if (agents[i].is_cooperator == 1) {
            for (int j = 0; j < NUM_AGENTS; j++) {
              if (link_matrix[i][j] == 1) {
                received_ro[j] += benefit;  // jが利得をもらう
                received_ro[i] -= cost;     // iがコストを払う
              }
            }
          }
        }  // 投資ゲーム終了
        for (int i = 0; i < NUM_AGENTS; i++) {
          if (link_count[i] > 0) {
            payoff_ro[i] = received_ro[i] / link_count[i];
          } else {
            payoff_ro[i] = 0.0f;
          }
        }  // リンク数割り利得計算終了

        for (int i = 0; i < NUM_AGENTS; i++) {
          count_game_ge[i] += 1;  // ゲーム数
          count_coop_game_ge[i] +=
              agents[i].is_cooperator;  // 協力者なら1、非協力者なら0
          count_poff_ge[i] += payoff_ro[i];
        }  // 世代での ゲーム数、協力したゲーム数、利得 追加計算終了

        float coop_game_rate[NUM_AGENTS];
        for (int i = 0; i < NUM_AGENTS; i++) {
          if (count_game_ge[i] > 0) {
            coop_game_rate[i] = (float)count_coop_game_ge[i] / count_game_ge[i];
          } else {
            coop_game_rate[i] =
                0.0f;  // ゲーム未参加者は0とする,今回は分母が0になることはなし
          }
        }  // 協力ゲーム率計算終了

        // WORK回、ランダムにエージェントペアを作ってリンクを更新
        for (int w = 0; w < work; w++) {
          int i = genrand_int32() % NUM_AGENTS;
          int j = genrand_int32() % NUM_AGENTS;
          if (i == j) continue;  // 同じエージェント同士はスキップ

          if (MODEL_TYPE == 1) {
            // モデルB: 「張る条件を満たした場合のみ」→ 張ってからランダムに切る
            if (link_matrix[i][j] == 0 && (coop_game_rate[i] >= agents[j].tf ||
                                           coop_game_rate[j] >= agents[i].tf)) {
              // リンクを張る
              link_matrix[i][j] = 1;
              link_matrix[j][i] = 1;
              // ランダムな既存リンクペアを探して切る
              while (1) {
                int u = genrand_int32() % NUM_AGENTS;
                int v = genrand_int32() % NUM_AGENTS;
                if (u != v && link_matrix[u][v] == 1) {
                  link_matrix[u][v] = 0;
                  link_matrix[v][u] = 0;
                  break;
                }
              }
            }
          }

        }  // ネットワーク切り貼り終了
      }  // ro終了
      int final_link_count[NUM_AGENTS] = {0};
      for (int i = 0; i < NUM_AGENTS; i++) {
        for (int j = 0; j < NUM_AGENTS; j++) {
          if (link_matrix[i][j] == 1) {
            final_link_count[i]++;
          }
        }
      }  // 全ラウンド終了後リンク数調べ終了、グラフ化のため

      // 社会学習 or 突然変異の意思決定を事前に記録
      int will_learn[NUM_AGENTS];  // 社会学習をする人はwill_learnが1
      for (int i = 0; i < NUM_AGENTS; i++) {
        float r_will = (float)genrand_real1();      // [0,1]
        will_learn[i] = (r_will < 1.0 - mutation);  // 0.99 の確率で社会学習
      }
      // 社会学習をするエージェントの順序をランダムに
      int indices[NUM_AGENTS];
      for (int i = 0; i < NUM_AGENTS; i++) indices[i] = i;
      shuffle(indices, NUM_AGENTS);
      // 社会学習を先にまとめて処理
      for (int k = 0; k < NUM_AGENTS; k++) {
        int i = indices[k];
        if (will_learn[i] == 0) continue;  // 突然変異勢はまずは無視
        // リンク相手の候補を収集
        int partners[NUM_AGENTS];
        int count = 0;
        for (int j = 0; j < NUM_AGENTS; j++) {
          if (link_matrix[i][j] == 1 && i != j) {
            partners[count++] = j;
          }
        }
        // リンクがないなら処理なし
        if (count == 0) {
          continue;
        }
        // ランダムに一人選んで模倣対象に
        int j = partners[genrand_int32() % count];
        // フェルミ関数により模倣確率を計算
        float diff = count_poff_ge[j] - count_poff_ge[i];
        float prob = 1.0f / (1.0f + expf(-beta * diff));
        float r_tc = (float)genrand_real1();  // [0,1]
        if (r_tc < prob) {
          agents[i].tc = agents[j].tc;
        }
        float r_tf = (float)genrand_real1();  // [0,1]
        if (r_tf < prob) {
          agents[i].tf = agents[j].tf;
        }
      }  // 社会学習終了
      // 突然変異（リンクなし社会学習含む）
      for (int i = 0; i < NUM_AGENTS; i++) {
        if (will_learn[i] == 1) continue;
        // tc, tl, tf のそれぞれを ±0.1 変化させる
        float *traits[2] = {&agents[i].tc, &agents[i].tf};
        for (int t = 0; t < 2; t++) {
          float delta = ((genrand_int32() % 2 == 0) ? 0.1f : -0.1f);
          *traits[t] += delta;
          if (*traits[t] < 0.0f) *traits[t] = 0.0f;  // TODO:
          if (*traits[t] > 1.1f) *traits[t] = 1.1f;  // TODO:
        }
      }  // 突然変異終了

      // ここから、tr,ge,i,tc,tl,tf,link数を行としたcsvを出力したい。
      // ---- CSV出力部（世代ごとに都度書き込む） ----
      if (tr == 0 && ge == 0) {
        fprintf(csv_file, "Trial,Generation,Agent,tc,tf,link_count\n");
      }
      for (int i = 0; i < NUM_AGENTS; i++) {
        fprintf(csv_file, "%d,%d,%d,%.2f,%.2f,%d\n", tr + 1, ge + 1, i,
                agents[i].tc, agents[i].tf, final_link_count[i]);
      }

      // どこまで処理したか出力
      printf("  Trial %d - Generation %d completed.\n", tr, ge);
    }  // ge終了
  }  // tr終了

  // ファイルを閉じる
  fclose(csv_file);
  printf("DONE\n");
}  // main終了

void part3() {
  // CSVファイルを出力するための準備 TODO:
  FILE *csv_file =
      fopen("3_keepd_01_form_or_t10_g10000_r100_w3000_b1.csv", "w");

  init_genrand((unsigned long)time(NULL));  // 乱数

  float density = 0.1f;  // [0.0, 1.0] 初期ネットワーク密度 // TODO:

  float benefit = 2.0f;    // 協力者が与える利得
  float cost = 1.0f;       // 協力にかかるコスト
  float beta = 1.0f;       // フェルミ関数の鋭さ TODO:
  float mutation = 0.01f;  // 突然変異確率

  int trial = 10;          // TODO:
  int generation = 10000;  // TODO:
  int round = 100;         // TODO:
  int work = 3000;         // TODO:

  for (int tr = 0; tr < trial; tr++) {  // トライアル開始
    printf("Trial %d:\n", tr);
    Agent agents[NUM_AGENTS];
    for (int i = 0; i < NUM_AGENTS; i++) {
      agents[i].tc = 0.0f;
      agents[i].tf = 0.0f;
      agents[i].is_cooperator = 0;
    }  // tc,tl,tf,の初期化 // 0スタート特有 TODO:

    // マトリクス初期化
    int link_matrix[NUM_AGENTS][NUM_AGENTS];
    memset(link_matrix, 0, sizeof(link_matrix));
    // 初期リンク密度によるリンク構築
    int max_possible_links = NUM_AGENTS * (NUM_AGENTS - 1) / 2;
    int target_link_count = (int)(density * max_possible_links);
    // 全てのリンク候補を生成（i < j のみ）
    int link_candidates[max_possible_links][2];
    int idx = 0;
    for (int i = 0; i < NUM_AGENTS; i++) {
      for (int j = i + 1; j < NUM_AGENTS; j++) {
        link_candidates[idx][0] = i;
        link_candidates[idx][1] = j;
        idx++;
      }
    }
    // ランダムにシャッフル
    for (int i = max_possible_links - 1; i > 0; i--) {
      int j = genrand_int32() % (i + 1);
      int temp0 = link_candidates[i][0];
      int temp1 = link_candidates[i][1];
      link_candidates[i][0] = link_candidates[j][0];
      link_candidates[i][1] = link_candidates[j][1];
      link_candidates[j][0] = temp0;
      link_candidates[j][1] = temp1;
    }
    // 最初の target_link_count 個を選んでリンクを張る
    for (int i = 0; i < target_link_count; i++) {
      int a = link_candidates[i][0];
      int b = link_candidates[i][1];
      link_matrix[a][b] = 1;
      link_matrix[b][a] = 1;  // 無向リンク
    }

    for (int ge = 0; ge < generation; ge++) {  // 世代開始
      int count_game_ge[NUM_AGENTS] = {0};
      int count_coop_game_ge[NUM_AGENTS] = {0};
      float count_poff_ge[NUM_AGENTS] = {0.0f};

      for (int ro = 0; ro < round; ro++) {  // ラウンド開始
        int link_count[NUM_AGENTS] = {0};
        for (int i = 0; i < NUM_AGENTS; i++) {
          for (int j = 0; j < NUM_AGENTS; j++) {
            if (link_matrix[i][j] == 1) {
              link_count[i]++;
            }
          }
        }  // リンク数調べ終了

        if (ro == 0) {
          for (int i = 0; i < NUM_AGENTS; i++) {
            float r_0rocd = (float)genrand_real1();  // [0,1]
            if (agents[i].tc <= r_0rocd) {
              agents[i].is_cooperator = 1;
            } else {
              agents[i].is_cooperator = 0;
            }
          }  // 協力非協力決め終了
        }  // 初回ラウンドは乱数で決める
        if (ro > 0) {
          for (int i = 0; i < NUM_AGENTS; i++) {
            if (link_count[i] > 0) {
              int coop_link_count[NUM_AGENTS] = {0};
              for (int i = 0; i < NUM_AGENTS; i++) {
                for (int j = 0; j < NUM_AGENTS; j++) {
                  if (link_matrix[i][j] == 1 && agents[j].is_cooperator == 1) {
                    coop_link_count[i]++;
                  }
                }
              }  // リンク協力者数数え終了（初回ラウンドは要らない）
              float coop_ratio = (float)coop_link_count[i] / link_count[i];
              if (agents[i].tc <= coop_ratio) {
                agents[i].is_cooperator = 1;
              } else {
                agents[i].is_cooperator = 0;
              }
            } else {  // リンクがない者は乱数と比較
              float r_nolcd = (float)genrand_real1();  // [0,1]
              if (agents[i].tc <= r_nolcd) {
                agents[i].is_cooperator = 1;
              } else {
                agents[i].is_cooperator = 0;
              }
            }
          }  // 協力非協力決め終了
        }  // 初回ラウンド以外

        float received_ro[NUM_AGENTS] = {
            0.0f};  // 各エージェントが受け取った利得
        float payoff_ro[NUM_AGENTS] = {
            0.0f};  // 各エージェントの最終利得（リンク数割り）
        // 協力者が利得を配り、自分はコストを負担
        for (int i = 0; i < NUM_AGENTS; i++) {
          if (agents[i].is_cooperator == 1) {
            for (int j = 0; j < NUM_AGENTS; j++) {
              if (link_matrix[i][j] == 1) {
                received_ro[j] += benefit;  // jが利得をもらう
                received_ro[i] -= cost;     // iがコストを払う
              }
            }
          }
        }  // 投資ゲーム終了
        for (int i = 0; i < NUM_AGENTS; i++) {
          if (link_count[i] > 0) {
            payoff_ro[i] = received_ro[i] / link_count[i];
          } else {
            payoff_ro[i] = 0.0f;
          }
        }  // リンク数割り利得計算終了

        for (int i = 0; i < NUM_AGENTS; i++) {
          count_game_ge[i] += 1;  // ゲーム数
          count_coop_game_ge[i] +=
              agents[i].is_cooperator;  // 協力者なら1、非協力者なら0
          count_poff_ge[i] += payoff_ro[i];
        }  // 世代での ゲーム数、協力したゲーム数、利得 追加計算終了

        float coop_game_rate[NUM_AGENTS];
        for (int i = 0; i < NUM_AGENTS; i++) {
          if (count_game_ge[i] > 0) {
            coop_game_rate[i] = (float)count_coop_game_ge[i] / count_game_ge[i];
          } else {
            coop_game_rate[i] =
                0.0f;  // ゲーム未参加者は0とする,今回は分母が0になることはなし
          }
        }  // 協力ゲーム率計算終了

        // WORK回、ランダムにエージェントペアを作ってリンクを更新
        for (int w = 0; w < work; w++) {
          int i = genrand_int32() % NUM_AGENTS;
          int j = genrand_int32() % NUM_AGENTS;
          if (i == j) continue;  // 同じエージェント同士はスキップ

          if (MODEL_TYPE == 1) {
            // モデルB: 「張る条件を満たした場合のみ」→ 張ってからランダムに切る
            if (link_matrix[i][j] == 0 && (coop_game_rate[i] >= agents[j].tf ||
                                           coop_game_rate[j] >= agents[i].tf)) {
              // リンクを張る
              link_matrix[i][j] = 1;
              link_matrix[j][i] = 1;
              // ランダムな既存リンクペアを探して切る
              while (1) {
                int u = genrand_int32() % NUM_AGENTS;
                int v = genrand_int32() % NUM_AGENTS;
                if (u != v && link_matrix[u][v] == 1) {
                  link_matrix[u][v] = 0;
                  link_matrix[v][u] = 0;
                  break;
                }
              }
            }
          }

        }  // ネットワーク切り貼り終了
      }  // ro終了
      int final_link_count[NUM_AGENTS] = {0};
      for (int i = 0; i < NUM_AGENTS; i++) {
        for (int j = 0; j < NUM_AGENTS; j++) {
          if (link_matrix[i][j] == 1) {
            final_link_count[i]++;
          }
        }
      }  // 全ラウンド終了後リンク数調べ終了、グラフ化のため

      // 社会学習 or 突然変異の意思決定を事前に記録
      int will_learn[NUM_AGENTS];  // 社会学習をする人はwill_learnが1
      for (int i = 0; i < NUM_AGENTS; i++) {
        float r_will = (float)genrand_real1();      // [0,1]
        will_learn[i] = (r_will < 1.0 - mutation);  // 0.99 の確率で社会学習
      }
      // 社会学習をするエージェントの順序をランダムに
      int indices[NUM_AGENTS];
      for (int i = 0; i < NUM_AGENTS; i++) indices[i] = i;
      shuffle(indices, NUM_AGENTS);
      // 社会学習を先にまとめて処理
      for (int k = 0; k < NUM_AGENTS; k++) {
        int i = indices[k];
        if (will_learn[i] == 0) continue;  // 突然変異勢はまずは無視
        // リンク相手の候補を収集
        int partners[NUM_AGENTS];
        int count = 0;
        for (int j = 0; j < NUM_AGENTS; j++) {
          if (link_matrix[i][j] == 1 && i != j) {
            partners[count++] = j;
          }
        }
        // リンクがないなら処理なし
        if (count == 0) {
          continue;
        }
        // ランダムに一人選んで模倣対象に
        int j = partners[genrand_int32() % count];
        // フェルミ関数により模倣確率を計算
        float diff = count_poff_ge[j] - count_poff_ge[i];
        float prob = 1.0f / (1.0f + expf(-beta * diff));
        float r_tc = (float)genrand_real1();  // [0,1]
        if (r_tc < prob) {
          agents[i].tc = agents[j].tc;
        }
        float r_tf = (float)genrand_real1();  // [0,1]
        if (r_tf < prob) {
          agents[i].tf = agents[j].tf;
        }
      }  // 社会学習終了
      // 突然変異（リンクなし社会学習含む）
      for (int i = 0; i < NUM_AGENTS; i++) {
        if (will_learn[i] == 1) continue;
        // tc, tl, tf のそれぞれを ±0.1 変化させる
        float *traits[2] = {&agents[i].tc, &agents[i].tf};
        for (int t = 0; t < 2; t++) {
          float delta = ((genrand_int32() % 2 == 0) ? 0.1f : -0.1f);
          *traits[t] += delta;
          if (*traits[t] < 0.0f) *traits[t] = 0.0f;  // TODO:
          if (*traits[t] > 1.1f) *traits[t] = 1.1f;  // TODO:
        }
      }  // 突然変異終了

      // ここから、tr,ge,i,tc,tl,tf,link数を行としたcsvを出力したい。
      // ---- CSV出力部（世代ごとに都度書き込む） ----
      if (tr == 0 && ge == 0) {
        fprintf(csv_file, "Trial,Generation,Agent,tc,tf,link_count\n");
      }
      for (int i = 0; i < NUM_AGENTS; i++) {
        fprintf(csv_file, "%d,%d,%d,%.2f,%.2f,%d\n", tr + 1, ge + 1, i,
                agents[i].tc, agents[i].tf, final_link_count[i]);
      }

      // どこまで処理したか出力
      printf("  Trial %d - Generation %d completed.\n", tr, ge);
    }  // ge終了
  }  // tr終了

  // ファイルを閉じる
  fclose(csv_file);
  printf("DONE\n");
}  // main終了

// main
int main() {
  part3();
  part5();
  return 0;
}