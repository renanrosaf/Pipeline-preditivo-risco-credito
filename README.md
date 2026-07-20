# Credit Risk Prediction — Pipeline Preditivo de Risco de Crédito

Pipeline preditivo de Machine Learning para classificação de risco de crédito (inadimplência vs. adimplência). Cobre EDA, tratamento de nulos/outliers, feature engineering (`comprometimento_renda`), balanceamento com SMOTE, escalonamento seguro, comparação KNN vs. Árvore de Decisão, análise de overfitting e veredito via matriz de confusão.

## 📌 Descrição do Problema de Negócio

No mercado bancário e de concessão de crédito, o lucro de uma instituição financeira depende do equilíbrio estratégico entre a arrecadação de juros e a mitigação da inadimplência. A concessão indiscriminada de empréstimos gera prejuízos líquidos severos por calotes, enquanto uma política excessivamente conservadora trava o crescimento da carteira e gera custos de oportunidade ao negar crédito a bons pagadores.

O objetivo deste projeto é desenvolver uma esteira analítica e um pipeline de Machine Learning de  para classificar os solicitantes de empréstimo entre **Adimplentes (Classe 0)** e **Inadimplentes (Classe 1)**, avaliando o impacto financeiro de dois tipos de erro:

- **Falso Positivo**: classificar um bom pagador como "Risco de Calote" → perda de receita (cliente rejeitado indevidamente).
- **Falso Negativo**: classificar um mau pagador como "Seguro" → prejuízo direto (empréstimo concedido e não pago).

A modelagem prioriza a redução dos **Falsos Negativos (Erro Tipo II)**, garantindo conformidade regulatória, interpretabilidade do risco e proteção direta ao caixa da instituição.

## 🎯 Objetivo do Modelo

Comparar dois algoritmos clássicos de classificação — **KNN** e **Árvore de Decisão**,  otimizando seus hiperparâmetros para evitar overfitting, e recomendar qual modelo deve ser colocado em produção com base na análise de negócio.

## 📁 Fonte dos Dados

O arquivo principal que alimenta o pipeline analítico está disponível publicamente em formato `.csv`:
- [Acessar base de dados no Google Drive](https://drive.google.com/file/d/1R4WCMc_56lv3fMaDalUBAz5jSxeZOcwI/view?usp=sharing)

## 📊 Dicionário de Dados

O conjunto de dados é composto por variáveis demográficas, financeiras e cadastrais dos clientes, tratadas via *Ordinal Encoding* e *One-Hot Encoding*:

| Variável | Tipo | Descrição |
| :--- | :--- | :--- |
| `person_age` | Numérica (Inteiro) | Idade do solicitante do empréstimo em anos. |
| `person_income` | Numérica (Contínua) | Renda anual total do cliente. |
| `person_emp_length` | Numérica (Contínua) | Tempo de estabilidade no emprego atual (em anos). |
| `loan_grade` | Numérica (Ordinal) | Classificação de risco do empréstimo (1 a 7, notas de A até G). |
| `loan_amnt` | Numérica (Contínua) | Valor total do empréstimo solicitado. |
| `loan_int_rate` | Numérica (Contínua) | Taxa de juros aplicada ao empréstimo (em %). |
| `loan_status` | Binária (Alvo - y) | Rótulo de inadimplência (0: Adimplente / 1: Inadimplente). |
| `loan_percent_income` | Numérica (Contínua) | Proporção do empréstimo em relação à renda anual. |
| `cb_person_cred_hist_length` | Numérica (Inteiro) | Tempo histórico de crédito do cliente nas bases de proteção (em anos). |
| `comprometimento_renda` | Numérica (Contínua) *(calculada)* | `(loan_amnt / person_income) * 100` — percentual da renda comprometido com o empréstimo. |
| `person_home_ownership_OWN` | Binária (0 ou 1) | Indica se o cliente possui imóvel próprio quitado. |
| `person_home_ownership_RENT` | Binária (0 ou 1) | Indica se o cliente reside em imóvel alugado. |
| `person_home_ownership_OTHER` | Binária (0 ou 1) | Outras situações de moradia (categoria base *MORTGAGE* removida por `drop_first`). |
| `loan_intent_EDUCATION` | Binária (0 ou 1) | Empréstimo destinado a financiamento educacional. |
| `loan_intent_HOMEIMPROVEMENT` | Binária (0 ou 1) | Empréstimo destinado a reformas ou melhorias no imóvel. |
| `loan_intent_MEDICAL` | Binária (0 ou 1) | Empréstimo destinado a despesas médicas de emergência ou tratamento. |
| `loan_intent_PERSONAL` | Binária (0 ou 1) | Empréstimo para fins pessoais gerais (categoria base *DEBTCONSOLIDATION* removida). |
| `loan_intent_VENTURE` | Binária (0 ou 1) | Empréstimo destinado ao aporte em novos empreendimentos comerciais. |
| `cb_person_default_on_file_Y` | Binária (0 ou 1) | Registro prévio de calote ou negativação no histórico dos órgãos de proteção. |

## 🗂️ Estrutura do Repositório

```text

├── notebooks/
│   └── pipeline_credito_risco.ipynb
├── data/
│   └── (dados brutos — não versionados, ver .gitignore)
├── docs/
│   └── (materiais de apoio, gráficos exportados, etc.)
├── requirements.txt
└── README.md
```

## ⚙️ Como Instalar as Dependências e Bibliotecas do Projeto

### Pré-requisitos
- Python 3.9 ou superior instalado no ambiente de desenvolvimento.
- Gerenciador de pacotes `pip` ou ambiente virtual (`venv` / `conda`).

### Passo a passo

```bash
# Clone o repositório
git clone https://github.com/renanrosaf/pipeline-preditivo-risco-credito.git
cd pipeline-preditivo-risco-credito

# Crie e ative o ambiente virtual
# Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# Windows:
python -m venv venv
venv\Scripts\activate

# Instale as dependências
pip install --upgrade pip
pip install -r requirements.txt
```

### Bibliotecas principais utilizadas e suas versões

| Biblioteca | Versão | Aplicação no Projeto |
| :--- | :---: | :--- |
| **pandas** | `3.0.3` | Manipulação de dados e engenharia de features |
| **numpy** | `2.5.1` | Operações matemáticas e vetoriais |
| **matplotlib** | `3.11.0` | Plotagem e customização de gráficos |
| **seaborn** | `0.13.2` | Visualização estatística e Análise Exploratória |
| **scikit-learn** | `1.9.0` | Pré-processamento, KNN, Árvore e métricas |
| **scipy** | `1.18.0` | Funções estatísticas de apoio |
| **imbalanced-learn** | `0.14.2` | Balanceamento de classes via SMOTE |
| **ipykernel** | `7.3.0` | Kernel de execução Python |
| **jupyterlab** | `4.6.1` | Ambiente de desenvolvimento (IDE) |

### Execução do Notebook

```bash
jupyter notebook notebooks/pipeline_credito_risco.ipynb
```

## 🧭 Fases do Projeto

- [x] Fase 1 — Análise Exploratória de Dados (EDA)
- [x] Fase 2 — Tratamento e Limpeza (Data Prep)
- [x] Fase 3 — Feature Engineering (`comprometimento_renda`)
- [x] Fase 4 — Separação, Balanceamento e Escalonamento Seguro
- [x] Fase 5 — Modelagem e Validação (KNN vs Árvore de Decisão)
- [x] Fase 6 — Avaliação e Veredito de Negócios

## 📈 Resumo Executivo

### Principais Insights da Análise Exploratória (EDA)

**Poder preditivo dos volumes extremos:** na análise de outliers, constatou-se que solicitações de empréstimo de valores elevadíssimos (`loan_amnt`) não representam ruídos de coleta, mas sim um padrão comportamental legítimo associado a um salto na taxa de calote. A preservação desses dados (sem clipping nessa variável) foi vital para alimentar os modelos com sinais reais de risco financeiro.

**Comprometimento de renda como fator letal:** a variável `comprometimento_renda` atua como o principal balizador de estrangulamento financeiro. Solicitantes que comprometem frações elevadas da renda anual apresentam alta correlação com notas de risco piores (`loan_grade` de D a G) e reincidência de calote (`cb_person_default_on_file_Y`).

**Desbalanceamento crítico da carteira:** a base original exibiu desbalanceamento de 78,13% de clientes bons contra 21,87% de inadimplentes. Para evitar que os algoritmos fossem enviesados pela maioria, o balanceamento por SMOTE foi aplicado exclusivamente na matriz de treino, preservando a proporção real do mercado no conjunto de teste (evitando *Data Leakage*).

### O Veredito: Por que a Árvore de Decisão venceu o KNN?

O projeto submeteu dois algoritmos de famílias distintas a uma rigorosa otimização de hiperparâmetros e monitoramento de overfitting. A comparação na amostra de teste consagrou a **Árvore de Decisão (`max_depth=7`)** como a solução definitiva para o deploy:

| Métrica / Critério | KNN (k=15) | Árvore de Decisão (max_depth=7) | Vencedor |
|---|---|---|---|
| F1-Score (Teste) | 66,10% | 75,07% | 🏆 Árvore (+8,97 pts) |
| Mitigação de Falsos Negativos | 492 calotes liberados | Redução drástica de calotes | 🏆 Árvore |
| Sensibilidade à escala | Exige padronização (StandardScaler) | Invariante à escala | 🏆 Árvore |
| Conformidade (Compliance) | Opaca (Black Box) | Explícita (White Box) | 🏆 Árvore |

A superioridade da Árvore de Decisão vem de sua mecânica de cortes ortogonais e monotônicos. Enquanto o KNN sofre com a maldição da dimensionalidade ao calcular distâncias euclidianas em 24 colunas, a Árvore segmenta os clientes diretamente nos limiares de dívida e histórico de crédito. Além disso, no setor bancário, a explicabilidade não é apenas um diferencial técnico, mas uma exigência regulatória (BACEN): a Árvore gera regras legíveis em formato "Se/Então", permitindo justificar legalmente a negação de crédito a qualquer cliente.

### Recomendação de Esteira Operacional

Para a implantação na API do banco, a probabilidade de inadimplência gerada pela Árvore (`predict_proba`) orienta um fluxo operacional automatizado em três esteiras:

- 🟢 **Risco Baixo** (probabilidade < 30%): aprovação instantânea na interface digital.
- 🟡 **Zona de Exceção** (probabilidade entre 30% e 65%): direcionamento para análise humana, com as três variáveis que mais impactaram a nota.
- 🔴 **Risco Crítico** (probabilidade > 65%): negação automática ou oferta restrita a linhas com garantia patrimonial real.

## 🎥 Vídeo de Apresentação

> 🚧 Link será adicionado após a gravação.

-- [Acessar link do video Google Drive](https://drive.google.com/file/d/1R4WCMc_56lv3fMaDalUBAz5jSxeZOcwI/view?usp=sharing)
*Projeto desenvolvido para o curso profissionalizante de Machine Learning e Visão Computacional [T2] — SCTEC-SENAI.*

## Autor

**Renan Rosa Ferreira**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/renanrosaferreira/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/renanrosaf)
