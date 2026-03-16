// only-leave model
// OR-leave
// w = 5000
// density = 10%

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mt19937ar.h"  // File for random number generation

#define NUM_AGENTS 100  // Number of agents

typedef struct {
  float tc;
  float tl;
  int is_cooperator;
} Agent;

// Utility for randomizing the order of agents
void shuffle(int* array, int n) {
  for (int i = n - 1; i > 0; i--) {
    int j = genrand_int32() % (i + 1);
    int temp = array[i];
    array[i] = array[j];
    array[j] = temp;
  }
}

#define MODEL_TYPE 0  // 0: only-leave model, 1: only-form model

void part01() {
  FILE* csv_file =
      fopen("3_keepd_01_leave_or_t10_g10000_r100_w5000_b1.csv", "w");

  init_genrand((unsigned long)time(NULL));

  float density = 0.1f;  // [0.0, 1.0] Initial network density

  float benefit = 2.0f;    // Benefit
  float cost = 1.0f;       // Cost
  float beta = 1.0f;       // Sharpness of the Fermi function
  float mutation = 0.01f;  // Mutation rate

  int trial = 10;          // Number of trial
  int generation = 10000;  // Number of generation
  int round = 100;         // Number of round
  int work = 5000;         // Number of work (network reconfigurations)

  for (int tr = 0; tr < trial; tr++) {  // Trial start
    printf("Trial %d:\n", tr);
    Agent agents[NUM_AGENTS];
    for (int i = 0; i < NUM_AGENTS; i++) {
      agents[i].tc = 0.0f;          // 0.0 (Cooperator) ~ 1.1 (Defector)
      agents[i].tl = 0.0f;          // 0.0 (Hard to leave) ~ 1.1 (Easy to leave)
      agents[i].is_cooperator = 0;  // Cooperator=1, Defector=0
    }  // Initialize tc,tl,tf

    // initialize the matrix as follows
    int link_matrix[NUM_AGENTS][NUM_AGENTS];
    memset(link_matrix, 0, sizeof(link_matrix));
    // Construct links based on initial link density
    int max_possible_links = NUM_AGENTS * (NUM_AGENTS - 1) / 2;
    int target_link_count = (int)(density * max_possible_links);
    // Generate all link candidates (only for i < j)
    int link_candidates[max_possible_links][2];
    int idx = 0;
    for (int i = 0; i < NUM_AGENTS; i++) {
      for (int j = i + 1; j < NUM_AGENTS; j++) {
        link_candidates[idx][0] = i;
        link_candidates[idx][1] = j;
        idx++;
      }
    }
    // Shuffle randomly
    for (int i = max_possible_links - 1; i > 0; i--) {
      int j = genrand_int32() % (i + 1);
      int temp0 = link_candidates[i][0];
      int temp1 = link_candidates[i][1];
      link_candidates[i][0] = link_candidates[j][0];
      link_candidates[i][1] = link_candidates[j][1];
      link_candidates[j][0] = temp0;
      link_candidates[j][1] = temp1;
    }
    // Select the target_link_count items and create links
    for (int i = 0; i < target_link_count; i++) {
      int a = link_candidates[i][0];
      int b = link_candidates[i][1];
      link_matrix[a][b] = 1;
      link_matrix[b][a] = 1;
    }

    for (int ge = 0; ge < generation; ge++) {  // Generation start
      int count_game_ge[NUM_AGENTS] = {0};
      int count_coop_game_ge[NUM_AGENTS] = {0};
      float count_poff_ge[NUM_AGENTS] = {0.0f};

      for (int ro = 0; ro < round; ro++) {  // Round start
        int link_count[NUM_AGENTS] = {0};
        for (int i = 0; i < NUM_AGENTS; i++) {
          for (int j = 0; j < NUM_AGENTS; j++) {
            if (link_matrix[i][j] == 1) {
              link_count[i]++;
            }
          }
        }  // Count the number of links

        if (ro == 0) {
          for (int i = 0; i < NUM_AGENTS; i++) {
            float r_0rocd = (float)genrand_real1();  // [0,1]
            if (agents[i].tc / 1.1 < r_0rocd) {
              agents[i].is_cooperator = 1;
            } else {
              agents[i].is_cooperator = 0;
            }
          }  // Determine whether an agent is a cooperator of a defector
        }  // If it is the first round, decide randomly
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
              }  // Count the number of linked cooperators
              float coop_ratio = (float)coop_link_count[i] / link_count[i];
              if (agents[i].tc <= coop_ratio) {
                agents[i].is_cooperator = 1;
              } else {
                agents[i].is_cooperator = 0;
              }
            } else {  // Compare with a random number for agents with no links
              float r_nolcd = (float)genrand_real1();  // [0,1]
              if (agents[i].tc <= r_nolcd) {
                agents[i].is_cooperator = 1;
              } else {
                agents[i].is_cooperator = 0;
              }
            }
          }  // Determine whether an agent is a cooperator of a defector
        }  // Except for the first round

        float received_ro[NUM_AGENTS] = {
            0.0f};  // Payoff received by each agent
        float payoff_ro[NUM_AGENTS] = {
            0.0f};  // Final payoff of each agent (devided by the number of
                    // links)
        for (int i = 0; i < NUM_AGENTS; i++) {
          if (agents[i].is_cooperator == 1) {
            for (int j = 0; j < NUM_AGENTS; j++) {
              if (link_matrix[i][j] == 1) {
                received_ro[j] += benefit;
                received_ro[i] -= cost;
              }
            }
          }
        }  // Game DONE
        for (int i = 0; i < NUM_AGENTS; i++) {
          if (link_count[i] > 0) {
            payoff_ro[i] = received_ro[i] / link_count[i];
          } else {
            payoff_ro[i] = 0.0f;
          }
        }  // Payoff calculation divided by the number of links

        for (int i = 0; i < NUM_AGENTS; i++) {
          count_game_ge[i] += 1;
          count_coop_game_ge[i] += agents[i].is_cooperator;
          count_poff_ge[i] += payoff_ro[i];
        }

        float coop_game_rate[NUM_AGENTS];
        for (int i = 0; i < NUM_AGENTS; i++) {
          if (count_game_ge[i] > 0) {
            coop_game_rate[i] = (float)count_coop_game_ge[i] / count_game_ge[i];
          } else {
            coop_game_rate[i] = 0.0f;
          }
        }  // Calculate the probability of cooperating in the games participated
           // in

        // For work iterations, randomly pair agents and update links
        for (int w = 0; w < work; w++) {
          int i = genrand_int32() % NUM_AGENTS;
          int j = genrand_int32() % NUM_AGENTS;
          if (i == j) continue;

          if (MODEL_TYPE == 0) { // Only-leave model
            // "Only if the cut conditioin is met"
            // → cut first, then create a link randomly
            if (link_matrix[i][j] == 1 &&
                (coop_game_rate[i] < agents[j].tl ||  // OR condition
                 coop_game_rate[j] < agents[i].tl)) {
              // leave
              link_matrix[i][j] = 0;
              link_matrix[j][i] = 0;
              // Find random unconnected pairs and create links
              while (1) {
                int u = genrand_int32() % NUM_AGENTS;
                int v = genrand_int32() % NUM_AGENTS;
                if (u != v && link_matrix[u][v] == 0) {
                  link_matrix[u][v] = 1;
                  link_matrix[v][u] = 1;
                  break;
                }
              }
            }
          }

        }  // End of Network rewiring
      }  // End of round
      int final_link_count[NUM_AGENTS] = {0};
      for (int i = 0; i < NUM_AGENTS; i++) {
        for (int j = 0; j < NUM_AGENTS; j++) {
          if (link_matrix[i][j] == 1) {
            final_link_count[i]++;
          }
        }
      }  // After all rounds finish, check link count (for graphing)

      // Record the decision of social learning or mutation in advance
      int will_learn[NUM_AGENTS];  // will_learn is 1, do social learning
      for (int i = 0; i < NUM_AGENTS; i++) {
        float r_will = (float)genrand_real1();  // [0,1]
        will_learn[i] =
            (r_will <
             1.0 - mutation);  // Social learning with a probability of 0.99
      }
      // Randomize the order of agents who perform social learning
      int indices[NUM_AGENTS];
      for (int i = 0; i < NUM_AGENTS; i++) indices[i] = i;
      shuffle(indices, NUM_AGENTS);
      // Process social learning first in a batch
      for (int k = 0; k < NUM_AGENTS; k++) {
        int i = indices[k];
        if (will_learn[i] == 0) continue;  // Ignore mutation agents for now
        // Collect linked partners
        int partners[NUM_AGENTS];
        int count = 0;
        for (int j = 0; j < NUM_AGENTS; j++) {
          if (link_matrix[i][j] == 1 && i != j) {
            partners[count++] = j;
          }
        }
        // Do nothing if there are no links
        if (count == 0) {
          continue;
        }
        // Randomly select one person as the imitation target
        int j = partners[genrand_int32() % count];
        // Calculate imitation probability using the Fermi function
        float diff = count_poff_ge[j] - count_poff_ge[i];
        float prob = 1.0f / (1.0f + expf(-beta * diff));
        float r_tc = (float)genrand_real1();  // [0,1]
        if (r_tc < prob) {
          agents[i].tc = agents[j].tc;
        }
        float r_tl = (float)genrand_real1();  // [0,1]
        if (r_tl < prob) {
          agents[i].tl = agents[j].tl;
        }
      }  // End of social learning
      // Mutation (including social learning agents without links)
      for (int i = 0; i < NUM_AGENTS; i++) {
        if (will_learn[i] == 1) continue;
        // change each of tc, tl, tf by ±0.1
        float* traits[2] = {&agents[i].tc, &agents[i].tl};
        for (int t = 0; t < 2; t++) {
          float delta = ((genrand_int32() % 2 == 0) ? 0.1f : -0.1f);
          *traits[t] += delta;
          if (*traits[t] < 0.0f) *traits[t] = 0.0f;  // TODO:
          if (*traits[t] > 1.1f) *traits[t] = 1.1f;  // TODO:
        }
      }  // End of mutation

      // CSV output section (write each generation as it occurs)
      if (tr == 0 && ge == 0) {
        fprintf(csv_file, "Trial,Generation,Agent,tc,tl,link_count\n");
      }
      for (int i = 0; i < NUM_AGENTS; i++) {
        fprintf(csv_file, "%d,%d,%d,%.2f,%.2f,%d\n", tr + 1, ge + 1, i,
                agents[i].tc, agents[i].tl, final_link_count[i]);
      }

      printf("  Trial %d - Generation %d completed.\n", tr, ge);
    }  // End of generation
  }  // End of trial

  fclose(csv_file);
  printf("DONE\n");
}  // main DONE

int main() {
  part01();
  return 0;
}