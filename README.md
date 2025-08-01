# 💡 Detecção de Falhas em Sensores com LSTM - Implementação em Jupyter

## 📌 Descrição do Projeto

Este projeto tem como objetivo detectar falhas em sensores utilizando Redes Neurais Recorrentes (RNN), com foco em LSTM (Long Short-Term Memory). A abordagem segue a **Estratégia B - LSTM Compartilhada "um sensor por vez"**, permitindo a análise de múltiplos sensores sem a necessidade de re-treinamento do modelo para cada um.

O código foi desenvolvido inteiramente em um Jupyter Notebook, tornando o fluxo interativo e fácil de acompanhar, desde a preparação dos dados até a avaliação final do modelo.

---

## 🧠 Metodologia com LSTM

O pipeline é estruturado nas seguintes etapas:

### 1. Importação das Bibliotecas

Principais bibliotecas utilizadas:

- `numpy`, `pandas` — manipulação de dados
- `matplotlib`, `seaborn` — visualização
- `sklearn` — métricas e divisão de dados
- `tensorflow.keras` — construção e treinamento do modelo

### 2. Leitura dos Dados

Os dados são lidos a partir de arquivos `.csv` contendo registros temporais de sensores.

### 3. Pré-processamento

- Normalização dos dados com estatísticas globais (média e desvio padrão)
- Criação de janelas de 240 amostras com stride de 120 (50% de sobreposição)

### 4. Divisão Temporal dos Dados

- 70% treino
- 15% validação
- 15% teste  
  (Sem embaralhamento para preservar a ordem temporal)

### 5. Construção do Modelo LSTM

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout
from tensorflow.keras.regularizers import l2

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(240, 1)))
model.add(BatchNormalization())
model.add(LSTM(64, kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
```

### 6. Treinamento com Callbacks

- EarlyStopping (`patience=20`)
- ReduceLROnPlateau (`factor=0.5`, `patience=10`)
- ModelCheckpoint (melhor modelo salvo com base em F1-Score)

### 7. Avaliação

- Métricas: Accuracy, Precision, Recall, F1-Score
- Matrizes de confusão
- Análise por sensor
- Curvas ROC

---

## 🧪 Resultados Esperados

- Modelo final salvo: `lstm_sensor_fault_detection_final.keras`
- F1-Score médio: **0.80–0.90**
- Análise detalhada por sensor exibida graficamente

---

## 📦 Bibliotecas Necessárias

```bash
tensorflow>=2.10
numpy
pandas
matplotlib
seaborn
scikit-learn
```

### Instalação com pip

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

### Instalação com conda (recomendado)

```bash
conda create -n lstm-fault-detection python=3.9
conda activate lstm-fault-detection
conda install numpy pandas matplotlib seaborn scikit-learn
pip install tensorflow
```

---

## 🚀 Como Executar

Clone o repositório:

```bash
git clone https://github.com/joaopedroalcn/G3_ATVEXTRA.git
cd G3_ATVEXTRA/JUPYTER
```

Execute o notebook:

```bash
jupyter notebook "Cópia_de_series_tmp_finall.ipynb"
```

Siga célula por célula o fluxo:

- Importação
- Pré-processamento
- Modelagem
- Treinamento
- Avaliação

---

## 🧩 Organização do Projeto

```mathematica
G3_ATVEXTRA/
├── JUPYTER/
│   └── Cópia_de_series_tmp_finall.ipynb
├── README.md
├── requirements.txt
└── environment.yml (opcional)
```

---

## 📚 Créditos e Orientação

Projeto desenvolvido para a disciplina de **Fundamentos de Redes Neurais** – Engenharia da Computação – Universidade Federal do Maranhão.

**Professor:** Dr. Thales Levi Azevedo Valente
**Alunos:** Joao Pedro de Alcantara Lima e Juan Pablo
