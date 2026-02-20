# k2_pytorch_pipelines_och_modelltr-ning_lucas_lindh

## Experimentresultat

| Learning rate | Batch size | Epochs | Validation accuracy |
|---:|---:|---:|---:|
| 0.01   | 32 | 5  | 0.1000 |
| 0.0001 | 32 | 5  | 0.5330 |
| 0.001  | 32 | 5  | 0.6664 |
| 0.001  | 64 | 5  | 0.6514 |
| 0.001  | 16 | 5  | 0.6719 |
| 0.001  | 16 | 10 | **0.6875** |
| 0.001  | 16 | 15 | 0.6659 |

### Analys
- För hög learning rate (0.01) gav instabil träning och låg acc.
- Learning rate 0.001 fungerade bäst.
- Mindre batch size (16) gav bättre resultat än 32 och 64.
- 10 epochs gav bäst accuracy (0.6875).
- Vid 15 epochs sjönk accuracy något, vilket kan tyda på overfitting.