#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_AGENTS 3

typedef struct {
  float tc;
  float tl;
  float tf;
  int is_cooperator;
} Agent;

int main() {
  int trial = 2;
  for (int tr = 0; tr < trial; tr++) {
    printf("Trial %d:\n", tr);
    Agent agents[NUM_AGENTS];
    memset(agents, 0, sizeof(agents));  // 0スタート特有

    // // 動作確認
    // for (int i = 0; i < NUM_AGENTS; i++) {
    //   printf("Agent %d -> tc: %.1f, tl: %.1f, tf: %.1f\n", i, agents[i].tc,
    //          agents[i].tl, agents[i].tf);
    // }
    // printf("\n");  // 改行で見やすく

    int link_matrix[NUM_AGENTS][NUM_AGENTS];
    memset(link_matrix, 0,
           sizeof(link_matrix));  // nullスタート特有//リセットあり特有

    // for (int i = 0; i < NUM_AGENTS; i++) {
    //   for (int j = 0; j < NUM_AGENTS; j++) {
    //     printf("%d ", link_matrix[i][j]);
    //   }
    //   printf("\n");
    // }

    int generation = 2;
    for (int ge = 0; ge < generation; ge++) {
      int round = 2;
      for (int ro = 0; ro < round; ro++) {
        if (ro == 0)
          ;
        int link_count[NUM_AGENTS] = {0};
        for (int i = 0; i < NUM_AGENTS; i++) {
          for (int j = 0; j < NUM_AGENTS; j++) {
            if (link_matrix[i][j] == 1) {
              link_count[i]++;
            }
          }
        }  // リンク数調べ終了

        for (int i = 0; i < NUM_AGENTS; i++) {
          float normalized_tc = agents[i].tc / 1.1f;
          float r = (float)rand() / RAND_MAX;  // [0.0, 1.0) の乱数
          if (normalized_tc <= r) {
            agents[i].is_cooperator = 1;
          } else {
            agents[i].is_cooperator = 0;
          }
        }  // 協力非協力決め終了

        int coop_link_count[NUM_AGENTS] = {0};
        for (int i = 0; i < NUM_AGENTS; i++) {
          for (int j = 0; j < NUM_AGENTS; j++) {
            if (link_matrix[i][j] == 1 && agents[j].is_cooperator == 1) {
              coop_link_count[i]++;
            }
          }
        }  // リンク協力者数数え終了

        float benefit = 2.0f;  // 協力者が与える利得
        float cost = 1.0f;     // 協力にかかるコスト
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

        int count_game_ge[NUM_AGENTS] = {0};
        int count_coop_game_ge[NUM_AGENTS] = {0};
        float count_poff_ge[NUM_AGENTS] = {0.0f};
        for (int i = 0; i < NUM_AGENTS; i++) {
          count_game_ge[i] = 1;  // 全員ゲームに参加したので1
          count_coop_game_ge[i] =
              agents[i].is_cooperator;  // 協力者なら1、非協力者なら0
          count_poff_ge[i] = payoff_ro[i];
        }  // 世代でのゲーム数、協力したゲーム数、利得計算終了

        if (ro > 0)
          ;

      }  // ro終了
    }  // ge終了

  }  // tr終了
}  // main終了

// ・有向リンク無向リンク・利得の割り算いるか