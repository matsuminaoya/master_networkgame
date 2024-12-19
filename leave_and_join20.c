#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mt19937ar-cok.h"

#define LAT 5  // lattice size LAT

/*leave&join 利得をリンクで割って模倣の計算 リンク0グラフから*/

int main(void) {
  int X = 100;       /*  人数  */
  int h = 100;       /*ラウンド数*/
  int sedai = 10000; /*世代数*/

  int simu = 50; /*シミュレーション回数*/
  int simucounts;

  double tc;
  double tleave;
  double tjoin;

  int T;             /*世代数え用*/
  int n;             /*人数数え用*/
  int nn;            /*相手のプレイヤーの番号を表す*/
  int t;             /*ラウンド数え用*/
  int i;             /* 協力に関する閾をカウント */
  int j;             /* leaveに関する閾値をカウント*/
  double b = 2.0;    /* 協力で得られるポイント　*/
  int c = 1;         /* 協力にかかるコスト　*/
  double a = -1;     /*誤差関数の係数*/
  double myu = 0.01; /*突然変異*/
  int g = 3000;      /*leave,joinを行う回数*/

  int givekeep[X]; /*個人があるラウンドで協力するかどうか*/
  int givekeepgoukei[X];
  double ritoku[X];   /*利得*/
  double giveritu[X]; /*何人がgiveしてくれたか*/
  double num;
  int active[X][X]; /*各プレイヤー同士の繋がりがactiveかどうか*/
  int sumall[X]; /*繋がってるそのラウンドでgiveしたプレイヤー数合計
                    世代最初に初期化*/
  int keika;      /*ラウンド数え用*/
  double sumgive; /*相手プレイヤーのそのラウンドまでの合計give数*/
  int link[X];    /*何人とlinkを結んでいるか*/
  int hikaku;
  double ran;
  int count;
  int countg;
  int countgg;
  int playerA;
  int playerB;
  double ikiti[X][3];
  double newikiti[X][3];

  int sumlink;
  double avelink[sedai];
  double averitoku[X];

  int counta;

  init_genrand((unsigned int)time(NULL));  // initialization for mt19937ar-cok.h

  FILE *fp, *fp1, *fp2, *fp3, *fp4, *fp5, *fp6, *fp7;

  fp = fopen("tc.csv", "w");
  fclose(fp);
  fp1 = fopen("tleave.csv", "w");
  fclose(fp1);
  fp2 = fopen("tjoin.csv", "w");
  fclose(fp2);

  fp7 = fopen("avelink.csv", "w");
  fclose(fp7);

  fp = fopen("tc.csv", "a");
  fprintf(fp, "t_c\n");
  fclose(fp);

  fp1 = fopen("tleave.csv", "a");
  fprintf(fp1, "t_leave\n");
  fclose(fp1);

  fp2 = fopen("tjoin.csv", "a");
  fprintf(fp2, "t_join\n");
  fclose(fp2);

  fp7 = fopen("avelink.csv", "a");
  fprintf(fp7, "average_link\n");
  fclose(fp7);

  for (countgg = 1; countgg < 20; countgg++) {
    if (countgg < 10) {
      g = countgg * 100;
    } else {
      g = (countgg - 9) * 1000;
    }

    fp = fopen("tc.csv", "a");
    fprintf(fp, "g=%d\n", g);
    fclose(fp);

    fp1 = fopen("tleave.csv", "a");
    fprintf(fp1, "g=%d\n", g);
    fclose(fp1);

    fp2 = fopen("tjoin.csv", "a");
    fprintf(fp2, "g=%d\n", g);
    fclose(fp2);

    fp7 = fopen("avelink.csv", "a");
    fprintf(fp7, "g=%d\n", g);
    fclose(fp7);

    /*繰り返し*/

    for (simucounts = 0; simucounts < simu; simucounts++) {
      for (i = 0; i < X; i++) {
        for (j = 0; j < 3; j++) {
          num = (double)(genrand_int32() % 12);
          ikiti[i][j] = num / 10.0;
        }
      }

      for (T = 0; T < sedai; T++) {
        for (n = 0; n < X; n++) {
          link[n] = 0;
          givekeepgoukei[n] = 0;
          ritoku[n] = 0;
        }

        for (i = 0; i < X; i++) {
          for (j = 0; j < X; j++) {
            active[i][j] = 0;
          }
        }

        for (n = 0; n < X; n++) {
          active[n][n] = 0; /*自分同士は非activeにしておく*/
        }

        for (n = 0; n < X; n++) {
          sumall[n] = 0;
        }

        /*1回目は全員がランダムに協力を選択する*/

        for (n = 0; n < X; n++) {
          ran = genrand_real1();
          if (ran > ikiti[n][0]) {
            givekeep[n] = 1;
          } else {
            givekeep[n] = 0;
          }
        }

        for (n = 0; n < X; n++) {
          givekeepgoukei[n] += givekeep[n];
        }

        for (n = 0; n < X; n++) {
          for (nn = 0; nn < X; nn++) {
            if (active[n][nn] == 1) {
              sumall[n] += givekeep[nn];
            }
          }
        }

        for (n = 0; n < X; n++) {
          ritoku[n] += (double)(b * (sumall[n]) - c * givekeep[n] * (X - 1));
        }

        /*ラウンド後半　つながっていればleave,つながっていなければjoinを選択
         * g回行う*/

        for (countg = 0; countg < g; countg++) {
          playerA = genrand_int32() % X;
          playerB = genrand_int32() % X;

          if (playerA == playerB) {
            num = genrand_int32() % (X - 1);
            playerB = playerB + num;
            if (playerB >= X) {
              playerB -= X;
            }
          }

          if (active[playerA][playerB] == 1) {
            if (ikiti[playerA][1] > (double)givekeep[playerB] ||
                ikiti[playerB][1] > (double)givekeep[playerA]) {
              active[playerA][playerB] = 0;
              active[playerB][playerA] = 0;
            }
          } else {
            if (ikiti[playerA][2] <= (double)givekeep[playerB] &&
                ikiti[playerB][2] <= (double)givekeep[playerA]) {
              active[playerA][playerB] = 1;
              active[playerB][playerA] = 1;
            }
          }
        }

        for (n = 0; n < X; n++) {
          for (nn = 0; nn < X; nn++) {
            if (active[n][nn] == 1) {
              link[n] += 1;
            }
          }
        }

        /*2回目以降は閾値に基づいて協力を選択*/

        for (t = 1; t < h; t++) {
          for (n = 0; n < X; n++) {
            sumall[n] = 0;
          }

          /*ラウンド前半　閾値より周囲のgive率が高ければgive*/
          for (n = 0; n < X; n++) {
            for (nn = 0; nn < X; nn++) {
              if (active[n][nn] == 1) {
                sumall[n] += givekeep[nn];
              }
            }
          }

          for (n = 0; n < X; n++) {
            if (link[n] > 0) {
              giveritu[n] = (double)((sumall[n]) / link[n]);
              if (ikiti[n][0] <= giveritu[n]) {
                givekeep[n] = 1;
              } else {
                givekeep[n] = 0;
              }
            } else {
              giveritu[n] = 0.0;
              givekeep[n] = 0;
            }
          }

          for (n = 0; n < X; n++) {
            givekeepgoukei[n] += givekeep[n];
          }

          /*利得の計算*/
          for (n = 0; n < X; n++) {
            for (nn = 0; nn < X; nn++) {
              if (active[n][nn] == 1) {
                ritoku[n] += (double)b * givekeep[nn] - (double)c * givekeep[n];
              }
            }
          }

          /*ラウンド後半　つながっていればleave,つながっていなければjoinを選択
           * g回行う*/

          for (countg = 0; countg < g; countg++) {
            playerA = genrand_int32() % X;
            playerB = genrand_int32() % X;

            if (playerA == playerB) {
              num = genrand_int32() % (X - 1);
              playerB = playerB + num;
              if (playerB >= X) {
                playerB -= X;
              }
            }

            if (active[playerA][playerB] == 1) {
              if (ikiti[playerA][1] >
                      (givekeepgoukei[playerB] / (double)(t + 1)) ||
                  ikiti[playerB][1] >
                      (givekeepgoukei[playerA] / (double)(t + 1))) {
                active[playerA][playerB] = 0;
                active[playerB][playerA] = 0;
              }
            } else {
              if (ikiti[playerA][2] <=
                      (givekeepgoukei[playerB] / (double)(t + 1)) &&
                  ikiti[playerB][2] <=
                      (givekeepgoukei[playerA] / (double)(t + 1))) {
                active[playerA][playerB] = 1;
                active[playerB][playerA] = 1;
              }
            }
          }

          for (n = 0; n < X; n++) {
            link[n] = 0;
          }

          for (n = 0; n < X; n++) {
            for (nn = 0; nn < X; nn++) {
              if (active[n][nn] == 1) {
                link[n] += 1;
              }
            }
          }
        }

        for (n = 0; n < X; n++) {
          newikiti[n][0] = ikiti[n][0];
          newikiti[n][1] = ikiti[n][1];
          newikiti[n][2] = ikiti[n][2];
        }

        for (n = 0; n < X; n++) {
          if (link[n] == 0) {
            averitoku[n] = 0;
          } else {
            averitoku[n] = ritoku[n] / link[n];
          }
        }

        /*１協力の閾値の変更*/
        for (n = 0; n < X; n++) {
          hikaku = genrand_int32() % X;
          for (count = 0; count < X; count++) {
            if (active[n][hikaku] == 1) {
              ran = genrand_real1();
              if (ran < 1 / (1 + exp((double)a *
                                     (averitoku[hikaku] - averitoku[n])))) {
                newikiti[n][0] = ikiti[hikaku][0];
              }
              break;
            }
            hikaku += 1;
            if (hikaku >= X) {
              hikaku = 0;
            }
          }
        }

        /*２leaveの閾値の変更*/
        for (n = 0; n < X; n++) {
          hikaku = genrand_int32() % X;
          for (count = 0; count < X; count++) {
            if (active[n][hikaku] == 1) {
              ran = genrand_real1();
              if (ran < 1 / (1 + exp((double)a *
                                     (averitoku[hikaku] - averitoku[n])))) {
                newikiti[n][1] = ikiti[hikaku][1];
              }
              break;
            }
            hikaku += 1;
            if (hikaku >= X) {
              hikaku = 0;
            }
          }
        }

        /*３joinの閾値の変更*/
        for (n = 0; n < X; n++) {
          hikaku = genrand_int32() % X;
          for (count = 0; count < X; count++) {
            if (active[n][hikaku] == 1) {
              ran = genrand_real1();
              if (ran < 1 / (1 + exp((double)a *
                                     (averitoku[hikaku] - averitoku[n])))) {
                newikiti[n][2] = ikiti[hikaku][2];
              }
              break;
            }
            hikaku += 1;
            if (hikaku >= X) {
              hikaku = 0;
            }
          }
        }

        for (n = 0; n < X; n++) {
          ikiti[n][0] = newikiti[n][0];
          ikiti[n][1] = newikiti[n][1];
          ikiti[n][2] = newikiti[n][2];
        }

        /*突然変異*/
        for (n = 0; n < X; n++) {
          ran = genrand_real1();
          if (ran < myu) {
            if (ikiti[n][0] == 0.0) {
              ran = genrand_real1();
              if (ran > 0.5) {
                ikiti[n][0] = 0.1;
              }
            } else if (ikiti[n][0] == 1.1) {
              ran = genrand_real1();
              if (ran > 0.5) {
                ikiti[n][0] = 1.0;
              }
            } else if (0 < ikiti[n][0] < 1) {
              ran = genrand_real1();
              if (ran > 0.5) {
                ikiti[n][0] += 0.1;
              } else {
                ikiti[n][0] -= 0.1;
              }
            }
          }
        }

        for (n = 0; n < X; n++) {
          ran = genrand_real1();
          if (ran < myu) {
            if (ikiti[n][1] == 0.0) {
              ran = genrand_real1();
              if (ran > 0.5) {
                ikiti[n][1] = 0.1;
              }
            } else if (ikiti[n][1] == 1.1) {
              ran = genrand_real1();
              if (ran > 0.5) {
                ikiti[n][1] = 1.0;
              }
            } else if (0 < ikiti[n][1] < 1) {
              ran = genrand_real1();
              if (ran > 0.5) {
                ikiti[n][1] += 0.1;
              } else {
                ikiti[n][1] -= 0.1;
              }
            }
          }
        }

        for (n = 0; n < X; n++) {
          ran = genrand_real1();
          if (ran < myu) {
            if (ikiti[n][2] == 0.0) {
              ran = genrand_real1();
              if (ran > 0.5) {
                ikiti[n][2] = 0.1;
              }
            } else if (ikiti[n][2] == 1.1) {
              ran = genrand_real1();
              if (ran > 0.5) {
                ikiti[n][2] = 1.0;
              }
            } else if (0 < ikiti[n][2] < 1) {
              ran = genrand_real1();
              if (ran > 0.5) {
                ikiti[n][2] += 0.1;
              } else {
                ikiti[n][2] -= 0.1;
              }
            }
          }
        }

        sumlink = 0;

        for (n = 0; n < X; n++) {
          sumlink += link[n];
        }
        avelink[T] += (double)(sumlink) / (double)(X);
      }

      tc = 0;
      tleave = 0;
      tjoin = 0;

      for (n = 0; n < X; n++) {
        tc += ikiti[n][0];
        tleave += ikiti[n][1];
        tjoin += ikiti[n][2];
      }
      tc = tc / (double)X;
      tleave = tleave / (double)X;
      tjoin = tjoin / (double)X;

      fp = fopen("tc.csv", "a");
      fprintf(fp, "%f\n", tc);
      fclose(fp);

      fp1 = fopen("tleave.csv", "a");
      fprintf(fp1, "%f\n", tleave);
      fclose(fp1);

      fp2 = fopen("tjoin.csv", "a");
      fprintf(fp2, "%f\n", tjoin);
      fclose(fp2);
    }

    for (T = 0; T < sedai; T++) {
      avelink[T] = avelink[T] / simu;

      fp7 = fopen("avelink.csv", "a");
      fprintf(fp7, "%f\n", avelink[T]);
      fclose(fp7);
    }
  }

  printf("simulation is over");

  return 0;
}
