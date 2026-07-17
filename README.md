# pipeline-preditivo-risco-credito
Pipeline preditivo de Machine Learning para classificação de risco de crédito (inadimplência vs. adimplência). Cobre EDA, tratamento de nulos/outliers, feature engineering (comprometimento_renda), balanceamento com SMOTE, escalonamento seguro, comparação KNN vs. Árvore de Decisão, análise de overfitting e veredito via matriz de confusão.

# Credit Risk Prediction — Pipeline Preditivo de Risco de Crédito

## 📌 Descrição do Problema de Negócio

Um banco precisa prever se um cliente se tornará **inadimplente** (`loan_status = 1`) ou se **pagará o empréstimo em dia** (`loan_status = 0`), com base em dados cadastrais e financeiros do solicitante.

O objetivo deste projeto é construir um pipeline preditivo completo de Machine Learning que apoie a decisão de concessão de crédito, avaliando o impacto financeiro de dois tipos de erro:

- **Falso Positivo**: classificar um bom pagador como "Risco de Calote" → perda de receita (cliente rejeitado indevidamente).
- **Falso Negativo**: classificar um mau pagador como "Seguro" → prejuízo direto (empréstimo concedido e não pago).

## 🎯 Objetivo do Modelo

Comparar dois algoritmos clássicos de classificação — **KNN** e **Árvore de Decisão** — otimizando seus hiperparâmetros para evitar overfitting, e recomendar qual modelo deve ser colocado em produção com base na análise de negócio.

## 📊 Dicionário de Dados

| Coluna | Descrição |
|---|---|
| `person_income` | Renda anual do solicitante |
| `loan_amnt` | Valor solicitado do empréstimo |
| `loan_status` | Variável alvo (1 = inadimplente, 0 = pagou em dia) |
| `comprometimento_renda` | *(coluna calculada)* `(loan_amnt / person_income) * 100` — percentual da renda comprometido com o empréstimo |
| *(demais colunas serão detalhadas conforme a EDA)* | |

> ⚠️ Este dicionário será atualizado à medida que a análise exploratória avançar.

## 🗂️ Estrutura do Repositório

```
├── notebooks/
│   └── pipeline_credit_risk.ipynb
├── data/
│   └── (dados brutos — não versionados, ver .gitignore)
├── docs/
│   └── (materiais de apoio, gráficos exportados, etc.)
├── requirements.txt
└── README.md
```

## ⚙️ Como Instalar as Dependências

```bash
git clone https://github.com/<seu-usuario>/credit-risk-prediction-ml.git
cd credit-risk-prediction-ml
pip install -r requirements.txt
```
```
Bibliotecas principais utilizadas:
- pandas
- numpy
- matplotlib / seaborn
- scikit-learn
- imbalanced-learn (SMOTE)
```

## 🧭 Fases do Projeto

- [x] Fase 1 — Análise Exploratória de Dados (EDA)
- [x] Fase 2 — Tratamento e Limpeza (Data Prep)
- [x] Fase 3 — Feature Engineering (`comprometimento_renda`)
- [x] Fase 4 — Separação, Balanceamento e Escalonamento Seguro
- [ ] Fase 5 — Modelagem e Validação (KNN vs Árvore de Decisão)
- [ ] Fase 6 — Avaliação e Veredito de Negócios

## 📈 Resumo Executivo

> 🚧 **Em construção.** Esta seção será preenchida ao final do projeto com os principais insights da EDA e o veredito final sobre qual modelo (KNN ou Árvore de Decisão) deve ser colocado em produção.

## 🎥 Vídeo de Apresentação

> 🚧 Link será adicionado após a gravação.

---
*Projeto desenvolvido para o curso profissionalizante de Machine Learning e Visão Computacional [T2] — SCTEC-SENAI.*


## Autor

**Renan Rosa Ferreira**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/renanrosaferreira/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/renanrosaf)
