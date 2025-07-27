# Detección de Fallas en Sensores con LSTM Compartilhada

## Descripción del Proyecto

Este proyecto implementa un modelo LSTM para la detección de fallas en señales de series temporales colectadas de sensores, utilizando la **Estrategia B - LSTM Compartilhada "um sensor por vez"**. El modelo puede aceptar cualquier cantidad de sensores sin necesidad de re-entrenamiento.

### Características Principales

- **Estrategia B**: LSTM compartilhada que procesa un sensor por vez
- **Janela temporal**: 240 amostras con stride de 120 (50% overlap)
- **Divisão temporal**: 70% treino, 15% validação, 15% teste
- **Normalização**: Usando estatísticas globais do dataset
- **Métricas**: Accuracy, Precision, Recall, F1-Score
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

## Estructura del Proyecto

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
├── README.md                       # Este archivo
├── requirements.txt                # Dependencias
└── environment.yml                 # Ambiente conda
```

## Instalación y Configuración

### Opción 1: Usando Conda (Recomendado)

```bash
# Crear ambiente conda
conda env create -f environment.yml

# Activar ambiente
conda activate sensor-fault-detection

# Verificar instalación
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
```

### Opción 2: Usando Pip

```bash
# Crear ambiente virtual
python -m venv sensor-fault-detection
source sensor-fault-detection/bin/activate  # Linux/Mac
# o
sensor-fault-detection\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### Dependencias Principales

- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter

## Uso

### 1. Ejecutar el Notebook Principal

```bash
# Iniciar Jupyter
jupyter notebook

# Abrir LSTM_Sensor_Fault_Detection.ipynb
```

### 2. Ejecutar Célula por Célula

El notebook está organizado en secciones:

1. **Importação das Bibliotecas**: Configuración inicial
2. **Configurações e Parâmetros**: Definición de hiperparámetros
3. **Carregamento e Preparação dos Dados**: Carga y preprocesamiento
4. **Criação de Janelas Temporais**: Implementación de la estrategia B
5. **Preparação do Dataset TensorFlow**: Creación de datasets
6. **Construção do Modelo LSTM**: Arquitectura del modelo
7. **Callbacks e Treinamento**: Entrenamiento
8. **Avaliação e Resultados**: Evaluación del modelo
9. **Análise por Sensor**: Análisis detallado por sensor
10. **Relatório Final**: Conclusiones y resultados

### 3. Resultados Esperados

- **Modelo entrenado**: `best_model.keras`
- **Modelo final**: `lstm_sensor_fault_detection_final.keras`
- **Configuración**: `model_config.json`
- **Métricas típicas**:
  - Accuracy: ~0.85-0.95
  - Precision: ~0.80-0.90
  - Recall: ~0.75-0.85
  - F1-Score: ~0.80-0.90

## Arquitectura del Modelo

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

## Estrategia B - LSTM Compartilhada

### Ventajas
- **Simplicidad**: Fácil implementación y mantenimiento
- **Escalabilidad**: Latencia linear con número de sensores
- **Flexibilidad**: Acepta cualquier número de sensores sin re-entrenamiento
- **Generalización**: Pesos compartidos permiten transferencia entre sensores

### Limitaciones
- No captura interacciones automáticas entre sensores
- Requiere ingeniería de features para dependencias entre sensores
- Puede beneficiarse de mecanismos de attention

### Implementación
1. **Janelas por sensor**: Cada sensor genera ventanas temporales independientes
2. **Embaralhamento**: Ventanas de sensores distintos se mezclan en el batch
3. **Pesos compartidos**: Una sola red LSTM procesa todos los sensores
4. **Inferência**: Se realiza sensor por sensor con datos ordenados temporalmente

## Parámetros de Configuración

```python
WINDOW_SIZE = 240      # Tamaño de la ventana temporal
STRIDE = 120          # Paso entre ventanas (50% overlap)
BATCH_SIZE = 32       # Tamaño del batch
LEARNING_RATE = 0.001 # Tasa de aprendizaje
EPOCHS = 100          # Número máximo de épocas
```

## División de Datos

- **Treino**: 70% (temporal, sin embaralhamento)
- **Validação**: 15% (temporal, sin embaralhamento)
- **Teste**: 15% (temporal, sin embaralhamento)

## Callbacks Utilizados

- **EarlyStopping**: Patience=20, monitor='val_f1_score'
- **ReduceLROnPlateau**: Factor=0.5, patience=10
- **ModelCheckpoint**: Guarda el mejor modelo basado en F1-Score

## Reproducibilidad

Para garantizar resultados reproducibles:

1. **Seeds fijos**: El código incluye configuración de seeds aleatorios
2. **Divisão temporal**: Los datos se dividen temporalmente sin embaralhamento
3. **Configuración**: Todos los hiperparámetros están documentados
4. **Ambiente**: Usar las versiones exactas de las dependencias

## Troubleshooting

### Problemas Comunes

1. **Memoria insuficiente**:
   ```python
   # Reducir batch_size
   BATCH_SIZE = 16
   ```

2. **Overfitting**:
   ```python
   # Aumentar regularización
   kernel_regularizer=l2(0.02)
   ```

3. **Underfitting**:
   ```python
   # Aumentar complejidad del modelo
   LSTM(128)  # en lugar de LSTM(64)
   ```

### Logs y Debugging

El notebook incluye logs detallados para debugging:
- Forma de los datos en cada etapa
- Distribución de clases
- Métricas de entrenamiento
- Análisis por sensor

## Contribución

Para contribuir al proyecto:

1. Fork el repositorio
2. Crear una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Crear un Pull Request

## Licencia

Este proyecto está bajo la licencia MIT.

## Contacto

Para preguntas o soporte, contactar al equipo del curso de Fundamentos de Redes Neurais.

---

**Universidade Federal do Maranhão**  
**Professor Dr. Thales Levi Azevedo Valente**  
**Engenharia da Computação - ECPXXXX - Fundamentos de Redes Neurais** 