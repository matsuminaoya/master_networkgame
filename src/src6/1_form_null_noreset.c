#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUM_AGENTS 100  // TODO:

typedef struct {
  float tc;
  float tf;
  int is_cooperator;
} Agent;

// エージェントのインデックスをシャッフルするユーティリティ //
// 社会学習の処理順を変えるだけ問題なし
void shuffle(int *array, int n) {
  for (int i = n - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    int temp = array[i];
    array[i] = array[j];
    array[j] = temp;
  }
}

int main() {
  // CSVファイルを出力するための準備
  FILE *csv_file =
      fopen("D:\\master\\src6\\1_f_reset_t10_g10000_r100_w5000_b1.csv",
            "w");  // TODO:
  if (csv_file == NULL) {
    printf("Error opening file for writing.\n");
    return 1;
  }

  srand(time(NULL));
  float benefit = 2.0f;  // 協力者が与える利得
  float cost = 1.0f;     // 協力にかかるコスト
  float beta = 1.0f;     // フェルミ関数の鋭さ（調整可）//TODO:

  int trial = 10;          // TODO:
  int generation = 10000;  // TODO:
  int round = 100;         // TODO:
  int work = 5000;         // TODO:

  // malloc用に定数を変数に格納（可読性のため）
  int T = trial;
  int G = generation;
  int N = NUM_AGENTS;
  // ヒープに確保
  float ***tc_list = malloc(T * sizeof(float **));
  float ***tf_list = malloc(T * sizeof(float **));
  int ***link_count_list = malloc(T * sizeof(int **));
  for (int t = 0; t < T; t++) {
    tc_list[t] = malloc(G * sizeof(float *));
    tf_list[t] = malloc(G * sizeof(float *));
    link_count_list[t] = malloc(G * sizeof(int *));
    for (int g = 0; g < G; g++) {
      tc_list[t][g] = malloc(N * sizeof(float));
      tf_list[t][g] = malloc(N * sizeof(float));
      link_count_list[t][g] = malloc(N * sizeof(int));
    }
  }

  for (int tr = 0; tr < trial; tr++) {
    printf("Trial %d:\n", tr);
    Agent agents[NUM_AGENTS];
    for (int i = 0; i < NUM_AGENTS; i++) {
      agents[i].tc = 0.0f;
      agents[i].tf = 0.0f;
      agents[i].is_cooperator = 0;
    }  // tc,tl,tf,の初期化 // 0スタート特有//TODO:

    // // 動作確認
    // for (int i = 0; i < NUM_AGENTS; i++) {
    //   printf("Agent %d -> tc: %.1f, tl: %.1f, tf: %.1f\n", i, agents[i].tc,
    //          agents[i].tl, agents[i].tf);
    // }
    // printf("\n");  // 改行で見やすく

    int link_matrix[NUM_AGENTS][NUM_AGENTS];
    for (int i = 0; i < NUM_AGENTS; i++) {
      for (int j = 0; j < NUM_AGENTS; j++) {
        link_matrix[i][j] = 0;
      }
    }  // nullスタート特有//リセットあり特有//TODO:

    // for (int i = 0; i < NUM_AGENTS; i++) {
    //   for (int j = 0; j < NUM_AGENTS; j++) {
    //     printf("%d ", link_matrix[i][j]);
    //   }
    //   printf("\n");
    // }

    for (int ge = 0; ge < generation; ge++) {
      int count_game_ge[NUM_AGENTS] = {0};
      int count_coop_game_ge[NUM_AGENTS] = {0};
      float count_poff_ge[NUM_AGENTS] = {0.0f};

      for (int ro = 0; ro < round; ro++) {
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
            // float normalized_tc = agents[i].tc / 1.1f;
            float r = (float)rand() / RAND_MAX;  // [0.0, 1.0) の乱数
            // if (normalized_tc <= r) {
            if (agents[i].tc <= r) {
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
            } else {
              float r = (float)rand() / RAND_MAX;
              if (agents[i].tc <= r) {
                agents[i].is_cooperator = 1;
              } else {
                agents[i].is_cooperator = 0;
              }
            }
          }
        }  // 初回ラウンド以外の協力者決め終了

        float received_ro[NUM_AGENTS] = {
            0.0f};  // 各エージェントが受け取った利得
        float payoff_ro[NUM_AGENTS] = {
            0.0f};  // 各エージェントの最終利得（リンク平均）
        // 協力者が利得を配り、自分はコストを負担
        for (int i = 0; i < NUM_AGENTS; i++) {
          if (agents[i].is_cooperator == 1) {
            for (int j = 0; j < NUM_AGENTS; j++) {
              if (link_matrix[i][j] == 1) {
                received_ro[j] += benefit;  // jが利得をもらう
                received_ro[i] -= cost;     // iがコストを払う（累積）
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
          count_game_ge[i] += 1;  // 全員ゲームに参加したので1
          count_coop_game_ge[i] +=
              agents[i].is_cooperator;  // 協力者なら1、非協力者なら0
          count_poff_ge[i] += payoff_ro[i];
        }  // 世代でのゲーム数、協力したゲーム数、利得計算終了

        float coop_game_rate[NUM_AGENTS];
        for (int i = 0; i < NUM_AGENTS; i++) {
          if (count_game_ge[i] > 0) {
            coop_game_rate[i] = (float)count_coop_game_ge[i] / count_game_ge[i];
          } else {
            coop_game_rate[i] = 0.0f;  // ゲーム未参加者は0とする
          }
        }  // 協力ゲーム率計算終了
        // WORK回、ランダムにエージェントペアを作ってリンクを更新

        for (int w = 0; w < work; w++) {
          int i = rand() % NUM_AGENTS;
          int j = rand() % NUM_AGENTS;
          if (i == j) continue;  // 同じエージェント同士はスキップ
          if (link_matrix[i][j] == 0) {
            // リンクがない場合、条件によりリンクを張る
            if (coop_game_rate[i] >= agents[j].tf &&
                coop_game_rate[j] >= agents[i].tf) {
              link_matrix[i][j] = 1;
              link_matrix[j][i] = 1;  // 対称性
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
      }  // 全ラウンド終了後リンク数調べ終了、グラフ化のため（最終ラウンドでの切り貼りがな場合はいらない）

      // 社会学習 or 突然変異の意思決定を事前に記録
      int will_learn[NUM_AGENTS];
      for (int i = 0; i < NUM_AGENTS; i++) {
        float r = (float)rand() / RAND_MAX;
        will_learn[i] = (r < 0.99f);  // 0.99 の確率で社会学習
      }
      // 社会学習をするエージェントの順序をランダムに
      int indices[NUM_AGENTS];
      for (int i = 0; i < NUM_AGENTS; i++) indices[i] = i;
      shuffle(indices, NUM_AGENTS);
      // 社会学習を先にまとめて処理
      for (int k = 0; k < NUM_AGENTS; k++) {
        int i = indices[k];
        if (will_learn[i] == 0) continue;
        // リンク相手の候補を収集
        int partners[NUM_AGENTS];
        int count = 0;
        for (int j = 0; j < NUM_AGENTS; j++) {
          if (link_matrix[i][j] == 1 && i != j) {
            partners[count++] = j;
          }
        }
        // リンクがないなら突然変異扱いに回す（あとで処理）
        if (count == 0) {
          will_learn[i] = 0;
          continue;
        }
        // ランダムに一人選んで模倣対象に
        int j = partners[rand() % count];
        // フェルミ関数により模倣確率を計算
        float diff = count_poff_ge[j] - count_poff_ge[i];
        float prob = 1.0f / (1.0f + expf(-beta * diff));
        float r = (float)rand() / RAND_MAX;
        if (r < prob) {
          agents[i].tc = agents[j].tc;
          agents[i].tf = agents[j].tf;
        }
      }  // 社会学習終了
      // 突然変異（リンクなし社会学習含む）
      for (int i = 0; i < NUM_AGENTS; i++) {
        if (will_learn[i] == 1) continue;
        // tc, tl, tf のそれぞれを ±0.1 変化させる
        float *traits[2] = {&agents[i].tc, &agents[i].tf};
        for (int t = 0; t < 2; t++) {
          float delta = ((rand() % 2 == 0) ? 0.1f : -0.1f);
          *traits[t] += delta;
          if (*traits[t] < 0.0f) *traits[t] = 0.0f;  // 反射させた// TODO:
          if (*traits[t] > 1.1f) *traits[t] = 1.1f;  // 反射させた// TODO:
        }
      }  // 突然変異終了

      // 世代の状態を記録
      for (int i = 0; i < NUM_AGENTS; i++) {
        tc_list[tr][ge][i] = agents[i].tc;
        tf_list[tr][ge][i] = agents[i].tf;
        link_count_list[tr][ge][i] = final_link_count[i];
      }

      printf("  Trial %d - Generation %d completed.\n", tr,
             ge);  // どこまで処理したか出力
    }  // ge終了
  }  // tr終了

  // ここから、tr,ge,i,tc,tl,tf,link数を行としたcsvを出力したい。

  // CSVヘッダーを書き込む
  fprintf(csv_file, "Trial,Generation,Agent,tc,tf,link_count\n");  // TODO:

  // シミュレーション結果をCSVファイルに書き込む
  for (int tr = 0; tr < trial; tr++) {
    for (int ge = 0; ge < generation; ge++) {
      for (int i = 0; i < NUM_AGENTS; i++) {
        fprintf(csv_file, "%d,%d,%d,%.2f,%.2f,%d\n", tr + 1, ge + 1, i,
                tc_list[tr][ge][i], tf_list[tr][ge][i],
                link_count_list[tr][ge][i]);
      }  // TODO:
    }
  }

  // ファイルを閉じる
  fclose(csv_file);
  printf("DONE\n");

  // メモリ解放
  for (int t = 0; t < T; t++) {
    for (int g = 0; g < G; g++) {
      free(tc_list[t][g]);
      free(tf_list[t][g]);
      free(link_count_list[t][g]);
    }
    free(tc_list[t]);
    free(tf_list[t]);
    free(link_count_list[t]);
  }
  free(tc_list);
  free(tf_list);
  free(link_count_list);
}  // main終了

// ・有向リンク無向リンク・利得の割り算いるか・跳ね返りのほうがいいかも
// はじめ1.1で割る必要ないね 切り貼りは最後のラウンドでもやっていいよね
// 社会学習をランダムにした //出力のときはtr+1,ge+1

// あとで、記録する世代を1000世代ごとにするとか飛ばす,フェルミ関数のベータを10に