# API-only generation benchmark
Model: MiniMax-M2.7 via MiniMax direct endpoint
Duration: 95.8s

| Mode | Accuracy | Judge correct | Avg tokens | Sum input | Sum output | Avg wall |
|---|---:|---:|---:|---:|---:|---:|
| full | 91.7% | 11/12 | 1884 | 19968 | 2639 | 3598ms |
| drop | 0.0% | 0/12 | 376 | 1143 | 3374 | 4379ms |

Drop vs full total token delta: -18090
Drop vs full input token delta: -18825
Drop vs full avg wall delta: +781ms
