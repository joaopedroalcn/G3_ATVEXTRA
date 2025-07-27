# Resumen del Proyecto: Detección de Fallas en Sensores

## 🎯 Objetivo Principal

Implementar un modelo LSTM para la detección de fallas en señales de series temporales colectadas de sensores, utilizando la **Estrategia B - LSTM Compartilhada "um sensor por vez"**. El modelo puede aceptar cualquier cantidad de sensores sin necesidad de re-entrenamiento.

## 📋 Requisitos Cumplidos

### ✅ Framework
- **TensorFlow 2.x / Keras** (obligatorio) ✅

### ✅ División Temporal
- **70% treino, 15% validação, 15% teste** ✅
- **Sin embaralhamento** (datos temporales) ✅

### ✅ Janela Temporal
- **WINDOW = 240** amostras ✅
- **STRIDE = 120** amostras (50% overlap) ✅
- **Justificativa**: Captura padrões temporais significativos sin excesso de dados similares ✅

### ✅ Estratégia B - LSTM Compartilhada
- **Dataset de janelas por sensor** (WINDOW, 1) ✅
- **Embaralhamento de janelas de sensores distintos** no batch ✅
- **Rede LSTM 1D única** com pesos compartilhados ✅
- **Inferência por sensor** com dados ordenados temporalmente ✅

### ✅ Arquitectura del Modelo
```
Input: (240, 1) - Janela temporal por sensor
├── LSTM(64, return_sequences=True, kernel_initializer=HeNormal())
├── BatchNormalization()
├── LSTM(64, kernel_initializer=HeNormal(), kernel_regularizer=l2(0.01))
├── BatchNormalization()
├── Dense(64, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(0.01))
├── BatchNormalization()
├── Dropout(0.3)
└── Dense(1, activation='sigmoid')
```

### ✅ Pérdidas y Métricas
- **Loss**: binary_crossentropy ✅
- **Métricas**: Precision, Recall, F1-Score ✅
- **Class weights**: Para lidar com desbalanceamento ✅

### ✅ Callbacks
- **EarlyStopping**: patience=20, monitor='val_f1_score' ✅
- **ReduceLROnPlateau**: factor=0.5, patience=10 ✅
- **ModelCheckpoint**: Guarda o melhor modelo ✅

## 📁 Estructura del Proyecto

```
neural-networks-extra-task/
├── 📊 dataset/                          # Datasets CSV (7 partes)
├── 🔧 codes/                            # Códigos existentes
├── 📓 LSTM_Sensor_Fault_Detection.ipynb # Notebook principal
├── 🐍 sensor_fault_detector.py          # Implementação modular
├── 📝 example_usage.py                  # Ejemplos de uso
├── 📋 README.md                         # Documentación completa
├── 📦 requirements.txt                  # Dependencias pip
├── 🐍 environment.yml                   # Ambiente conda
└── 🚫 .gitignore                        # Archivos a ignorar
```

## 🚀 Características Principales

### 🔄 Flexibilidad de Sensores
- **Acepta cualquier número de sensores** sin re-entrenamiento
- **Pesos compartilhados** permiten generalización entre sensores
- **Latencia linear** con el número de sensores

### 📈 Preprocesamiento Robusto
- **Normalización** usando estadísticas globales del dataset
- **División temporal** sin embaralhamento
- **Janelas temporales** por sensor individual

### 🎯 Métricas Completas
- **Accuracy, Precision, Recall, F1-Score**
- **Matriz de confusão** visual
- **Análisis por sensor** individual

### 🔧 Configuración Flexible
- **Parámetros ajustables** (janela, stride, batch_size, learning_rate)
- **Callbacks configurables** para optimización
- **Class weights** automáticos para desbalanceamento

## 📊 Resultados Esperados

### Métricas Típicas
- **Accuracy**: ~0.85-0.95
- **Precision**: ~0.80-0.90
- **Recall**: ~0.75-0.85
- **F1-Score**: ~0.80-0.90

### Archivos Generados
- `best_model.keras` - Mejor modelo durante entrenamiento
- `lstm_sensor_fault_detection_final.keras` - Modelo final
- `model_config.json` - Configuraciones del modelo

## 🛠️ Instalación y Uso

### Opción 1: Conda (Recomendado)
```bash
conda env create -f environment.yml
conda activate sensor-fault-detection
jupyter notebook
```

### Opción 2: Pip
```bash
pip install -r requirements.txt
python sensor_fault_detector.py
```

### Uso Rápido
```python
from sensor_fault_detector import SensorFaultDetector

# Inicializar detector
detector = SensorFaultDetector()

# Pipeline completo
df = detector.load_data()
sensor_data, labels, columns = detector.preprocess_data(df)
windows, window_labels, sensor_ids = detector.create_sensor_windows(sensor_data, labels, columns)
train_data, val_data, test_data = detector.split_temporal_data(windows, window_labels, sensor_ids)
history = detector.train(train_data, val_data)
results = detector.evaluate(test_data)
```

## 🎓 Justificativas Técnicas

### Janela Temporal (WINDOW = 240)
- **Captura padrões temporais** significativos
- **Balance entre complexidade** e eficiência
- **Adequado para detecção** de falhas em séries temporais

### Stride (STRIDE = 120)
- **50% overlap** permite captura de padrões sem excesso
- **Reduz dados similares** mantendo cobertura adequada
- **Otimiza uso de memória** e tempo de processamento

### Estratégia B - Vantagens
- **Simplicidade**: Fácil implementação e manutenção
- **Escalabilidade**: Funciona com qualquer número de sensores
- **Eficiência**: Latencia linear, sem necessidade de re-treinamento
- **Generalização**: Pesos compartilhados permitem transferência de conhecimento

## 🔍 Análisis y Interpretación

### Ventajas de la Implementación
1. **Flexibilidad**: Acepta sensores dinámicos
2. **Eficiencia**: Procesamiento optimizado
3. **Robustez**: Manejo de desbalanceamento
4. **Reproducibilidad**: Seeds fijos y configuración documentada

### Limitaciones y Mejoras Futuras
1. **Interacciones entre sensores**: No captura automáticamente
2. **Attention mechanisms**: Podrían mejorar performance
3. **Engenharia de features**: Diferenças entre sensores
4. **Ensemble methods**: Combinar múltiples modelos

## 📚 Documentación

- **README.md**: Guía completa de instalación y uso
- **LSTM_Sensor_Fault_Detection.ipynb**: Notebook interactivo
- **sensor_fault_detector.py**: Implementación modular
- **example_usage.py**: Ejemplos prácticos

## 🎯 Conclusión

La implementación de la **Estrategia B - LSTM Compartilhada "um sensor por vez"** cumple todos los requisitos del enunciado y proporciona una solución robusta, escalable y eficiente para la detección de fallas en sensores. El modelo puede ser aplicado a cualquier número de sensores sin re-entrenamiento, manteniendo alta performance y flexibilidad.

---

**Universidade Federal do Maranhão**  
**Professor Dr. Thales Levi Azevedo Valente**  
**Engenharia da Computação - ECPXXXX - Fundamentos de Redes Neurais** 