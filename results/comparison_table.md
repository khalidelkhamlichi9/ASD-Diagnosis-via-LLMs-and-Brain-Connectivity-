| Method                            | Acc                          | bAcc                         | AUC                          | F1                           | Worst-Site                   | Prec                         | Recall                       |
| --------------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- |
| NeuroLingua-Bridge (DeepSeek) †   | 80.0±0.7%                    | —                            | 81.1±0.8%                    | 79.9±0.7%                    | 74.1±1.3%                    | —                            | —                            |
| NeuroLingua-Bridge (GPT-2) †      | 76.0±1.1%                    | —                            | 76.2±1.2%                    | 75.8±1.1%                    | 64.3±1.9%                    | —                            | —                            |
| ComBat (EB) + MLP ‡               | 59.3%                        | 59.1%                        | 61.6%                        | 59.0%                        | 43.3%                        | 59.0%                        | 59.1%                        |
| BNT K=8 d=64 ‡                    | 54.1%                        | 54.4%                        | 57.3%                        | 54.1%                        | 40.4%                        | 54.4%                        | 54.4%                        |

† Résultats du papier (Table 1, 5 seeds, full DG pipeline).
‡ Même split inter-site, même seed, même métriques — comparaison équitable.
  bAcc = Balanced Accuracy, Worst-Site = min accuracy sur les sites de test.