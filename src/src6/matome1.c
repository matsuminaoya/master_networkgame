#include <stdio.h>
#include <string.h>

#define NUM_AGENTS 3

typedef struct {
  float tc;
  float tl;
  float tf;
} Agent;

int main() {
  int trial = 2;

  for (int tr = 0; tr < trial; tr++) {
    printf("Trial %d:\n", tr);
    Agent agents[NUM_AGENTS];
    memset(agents, 0, sizeof(agents));

    // // 動作確認
    // for (int i = 0; i < NUM_AGENTS; i++) {
    //   printf("Agent %d -> tc: %.1f, tl: %.1f, tf: %.1f\n", i, agents[i].tc,
    //          agents[i].tl, agents[i].tf);
    // }
    // printf("\n");  // 改行で見やすく


  }  // for (int tr = 0; < trial; tr++)
}// int main()