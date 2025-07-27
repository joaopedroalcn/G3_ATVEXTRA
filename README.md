# Detecção de Falhas em Sensores com LSTM Compartilhada

## Descrição do Projeto

Este projeto implementa um modelo LSTM para a detecção de falhas em sinais de séries temporais coletadas de sensores, utilizando a **Estratégia B - LSTM Compartilhada "um sensor por vez"**. O modelo pode aceitar qualquer quantidade de sensores sem necessidade de re-treinamento.

### Características Principais

- **Estratégia B**: LSTM compartilhada que processa um sensor por vez
- **Janela temporal**: 240 amostras com stride de 120 (50% sobreposição)
- **Divisão temporal**: 70% treino, 15% validação, 15% teste
- **Normalização**: Usando estatísticas globais do dataset
- **Métricas**: Accuracy, Precision, Recall, F1-Score
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

## Estrutura do Projeto

```
neural-networks-extra-task/
├── codes/                          # Códigos existentes
│   ├── LSTM.py
│   ├── LSTMTrainingManager.py
│   ├── ModelEvaluator.py
│   ├── ModelMetricsManager.py
│   └── TensorflowDataPreprocessor.py
├── dataset/                        # Datasets CSV
│   ├── dataset_parte_1.csv
│   ├── dataset_parte_2.csv
│   ├── dataset_parte_3.csv
│   ├── dataset_parte_4.csv
│   ├── dataset_parte_5.csv
│   ├── dataset_parte_6.csv
│   └── dataset_parte_7.csv
├── LSTM_Sensor_Fault_Detection.ipynb  # Notebook principal
├── README.md                       # Este arquivo
├── requirements.txt                # Dependências
└── environment.yml                 # Ambiente conda
```

## Instalação e Configuração

### Opção 1: Usando Conda (Recomendado)

```bash
# Criar ambiente conda
conda env create -f environment.yml

# Ativar ambiente
conda activate sensor-fault-detection

# Verificar instalação
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
```

### Opção 2: Usando Pip

```bash
# Criar ambiente virtual
python -m venv sensor-fault-detection
source sensor-fault-detection/bin/activate  # Linux/Mac
# ou
sensor-fault-detection\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

### Dependências Principais

- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter

## Uso

### 1. Executar o Notebook Principal

```bash
# Iniciar Jupyter
jupyter notebook

# Abrir LSTM_Sensor_Fault_Detection.ipynb
```

### 2. Executar Célula por Célula

O notebook está organizado em seções:

1. **Importação das Bibliotecas**: Configuração inicial
2. **Configurações e Parâmetros**: Definição de hiperparâmetros
3. **Carregamento e Preparação dos Dados**: Carga e pré-processamento
4. **Criação de Janelas Temporais**: Implementação da estratégia B
5. **Preparação do Dataset TensorFlow**: Criação de datasets
6. **Construção do Modelo LSTM**: Arquitetura do modelo
7. **Callbacks e Treinamento**: Treinamento
8. **Avaliação e Resultados**: Avaliação do modelo
9. **Análise por Sensor**: Análise detalhada por sensor
10. **Relatório Final**: Conclusões e resultados

### 3. Resultados Esperados

- **Modelo treinado**: `best_model.keras`
- **Modelo final**: `lstm_sensor_fault_detection_final.keras`
- **Configuração**: `model_config.json`
- **Métricas típicas**:
  - Accuracy: ~0.85-0.95
  - Precision: ~0.80-0.90
  - Recall: ~0.75-0.85
  - F1-Score: ~0.80-0.90

## Arquitetura do Modelo

```
Input: (240, 1) - Janela temporal por sensor
├── LSTM(64, return_sequences=True)
├── BatchNormalization()
├── LSTM(64, kernel_regularizer=l2(0.01))
├── BatchNormalization()
├── Dense(64, activation='relu', kernel_regularizer=l2(0.01))
├── BatchNormalization()
├── Dropout(0.3)
└── Dense(1, activation='sigmoid')
```

## Estratégia B - LSTM Compartilhada

### Vantagens
- **Simplicidade**: Fácil implementação e manutenção
- **Escalabilidade**: Latência linear com número de sensores
- **Flexibilidade**: Aceita qualquer número de sensores sem re-treinamento
- **Generalização**: Pesos compartilhados permitem transferência entre sensores

### Limitações
- Não captura interações automáticas entre sensores
- Requer engenharia de features para dependências entre sensores
- Pode se beneficiar de mecanismos de attention

### Implementação
1. **Janelas por sensor**: Cada sensor gera janelas temporais independentes
2. **Embaralhamento**: Janelas de sensores distintos se misturam no batch
3. **Pesos compartilhados**: Uma única rede LSTM processa todos os sensores
4. **Inferência**: É realizada sensor por sensor com dados ordenados temporalmente

## Parâmetros de Configuração

```python
WINDOW_SIZE = 240      # Tamanho da janela temporal
STRIDE = 120          # Passo entre janelas (50% sobreposição)
BATCH_SIZE = 32       # Tamanho do batch
LEARNING_RATE = 0.001 # Taxa de aprendizado
EPOCHS = 100          # Número máximo de épocas
```

## Divisão de Dados

- **Treino**: 70% (temporal, sem embaralhamento)
- **Validação**: 15% (temporal, sem embaralhamento)
- **Teste**: 15% (temporal, sem embaralhamento)

## Callbacks Utilizados

- **EarlyStopping**: Patience=20, monitor='val_f1_score'
- **ReduceLROnPlateau**: Factor=0.5, patience=10
- **ModelCheckpoint**: Salva o melhor modelo baseado em F1-Score

## Reproducibilidade

Para garantir resultados reproduzíveis:

1. **Seeds fixos**: O código inclui configuração de seeds aleatórios
2. **Divisão temporal**: Os dados são divididos temporalmente sem embaralhamento
3. **Configuração**: Todos os hiperparâmetros estão documentados
4. **Ambiente**: Usar as versões exatas das dependências

## Solução de Problemas

### Problemas Comuns

1. **Memória insuficiente**:
   ```python
   # Reduzir batch_size
   BATCH_SIZE = 16
   ```

2. **Overfitting**:
   ```python
   # Aumentar regularização
   kernel_regularizer=l2(0.02)
   ```

3. **Underfitting**:
   ```python
   # Aumentar complexidade do modelo
   LSTM(128)  # em vez de LSTM(64)
   ```

### Logs e Debugging

O notebook inclui logs detalhados para debugging:
- Forma dos dados em cada etapa
- Distribuição de classes
- Métricas de treinamento
- Análise por sensor

## Contribuição

Para contribuir com o projeto:

1. Fork o repositório
2. Criar uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Criar um Pull Request

## Licença

Este projeto está sob a licença MIT.

## Contato

Para perguntas ou suporte, contactar a equipe do curso de Fundamentos de Redes Neurais.

---

**Universidade Federal do Maranhão**  
**Professor Dr. Thales Levi Azevedo Valente**  
**Engenharia da Computação - ECPXXXX - Fundamentos de Redes Neurais** 
