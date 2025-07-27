#!/usr/bin/env python3
"""
Sensor Fault Detection with Shared LSTM
Estratégia B: "Um sensor por vez"

Este script implementa um modelo LSTM compartilhado para detecção de falhas
em sinais de sensores, aceitando qualquer quantidade de sensores sem re-treinamento.

Autor: Implementação baseada no enunciado do curso
Universidade Federal do Maranhão
Professor Dr. Thales Levi Azevedo Valente
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class F1Score(keras.metrics.Metric):
    """Métrica F1-Score customizada"""
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_binary = tf.round(y_pred)
        self.precision.update_state(y_true, y_pred_binary, sample_weight)
        self.recall.update_state(y_true, y_pred_binary, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return tf.cond(
            tf.equal(p + r, 0),
            lambda: tf.constant(0.0),
            lambda: 2 * (p * r) / (p + r)
        )

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

class SensorFaultDetector:
    """
    Classe principal para detecção de falhas em sensores usando LSTM compartilhada
    """
    
    def __init__(self, window_size=240, stride=120, batch_size=32, learning_rate=0.001):
        """
        Inicializa o detector de falhas
        
        Args:
            window_size (int): Tamanho da janela temporal
            stride (int): Passo entre janelas
            batch_size (int): Tamanho do batch
            learning_rate (float): Taxa de aprendizagem
        """
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Estatísticas globais do dataset
        self.global_mean = 31.9291
        self.global_std = 16058.15
        
        # Modelo
        self.model = None
        self.class_weights = None
        
        # Configurar seeds para reprodutibilidade
        np.random.seed(42)
        tf.random.set_seed(42)
        
    def load_data(self, data_dir='dataset/'):
        """
        Carrega e combina todos os arquivos CSV do dataset
        
        Args:
            data_dir (str): Diretório com os arquivos CSV
            
        Returns:
            pd.DataFrame: Dataset combinado
        """
        csv_files = glob.glob(os.path.join(data_dir, 'dataset_parte_*.csv'))
        csv_files.sort()
        
        print(f"Carregando {len(csv_files)} arquivos CSV...")
        
        dataframes = []
        for file in csv_files:
            df = pd.read_csv(file)
            dataframes.append(df)
            print(f"  {file}: {df.shape[0]} linhas, {df.shape[1]} colunas")
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"Dataset combinado: {combined_df.shape[0]} linhas, {combined_df.shape[1]} colunas")
        
        return combined_df
    
    def preprocess_data(self, df):
        """
        Pré-processa os dados: normalização e preparação das labels
        
        Args:
            df (pd.DataFrame): Dataset bruto
            
        Returns:
            tuple: (dados_normalizados, labels_binárias, colunas_sensores)
        """
        # Separar features (sensores) e labels
        sensor_columns = [col for col in df.columns if col.startswith('sensor_')]
        
        # Dados dos sensores
        sensor_data = df[sensor_columns].values
        
        # Normalização usando as estatísticas globais
        sensor_data_normalized = (sensor_data - self.global_mean) / self.global_std
        
        # Preparar labels
        labels = df['label'].fillna('').values
        binary_labels = np.array([1 if label.strip() != '' else 0 for label in labels])
        
        print(f"Dados normalizados:")
        print(f"- Shape dos dados: {sensor_data_normalized.shape}")
        print(f"- Número de sensores: {len(sensor_columns)}")
        print(f"- Distribuição das labels: {np.bincount(binary_labels)}")
        
        return sensor_data_normalized, binary_labels, sensor_columns
    
    def create_sensor_windows(self, sensor_data, labels, sensor_columns):
        """
        Cria janelas temporais para cada sensor individualmente (Estratégia B)
        
        Args:
            sensor_data (np.array): Dados dos sensores normalizados
            labels (np.array): Labels binárias
            sensor_columns (list): Lista de nomes dos sensores
            
        Returns:
            tuple: (janelas, labels_janelas, ids_sensores)
        """
        windows = []
        window_labels = []
        sensor_ids = []
        
        num_sensors = len(sensor_columns)
        num_timestamps = sensor_data.shape[0]
        
        print(f"Criando janelas para {num_sensors} sensores...")
        
        for sensor_idx in range(num_sensors):
            sensor_signal = sensor_data[:, sensor_idx]
            sensor_label = labels
            
            # Criar janelas para este sensor
            for start_idx in range(0, num_timestamps - self.window_size + 1, self.stride):
                end_idx = start_idx + self.window_size
                
                # Janela de dados do sensor
                window = sensor_signal[start_idx:end_idx]
                
                # Label da janela (1 se há falha em qualquer ponto da janela)
                window_label = 1 if np.any(sensor_label[start_idx:end_idx] == 1) else 0
                
                windows.append(window)
                window_labels.append(window_label)
                sensor_ids.append(sensor_idx)
            
            if sensor_idx % 10 == 0:
                print(f"  Sensor {sensor_idx + 1}/{num_sensors}: {len([w for w in window_labels if w == 1])} janelas com falha")
        
        windows = np.array(windows)
        window_labels = np.array(window_labels)
        sensor_ids = np.array(sensor_ids)
        
        print(f"Janelas criadas:")
        print(f"- Total de janelas: {len(windows)}")
        print(f"- Shape das janelas: {windows.shape}")
        print(f"- Distribuição das labels: {np.bincount(window_labels)}")
        
        return windows, window_labels, sensor_ids
    
    def split_temporal_data(self, windows, window_labels, sensor_ids, train_split=0.7, val_split=0.15):
        """
        Divide os dados temporalmente (sem embaralhar)
        
        Args:
            windows (np.array): Janelas de dados
            window_labels (np.array): Labels das janelas
            sensor_ids (np.array): IDs dos sensores
            train_split (float): Proporção para treino
            val_split (float): Proporção para validação
            
        Returns:
            tuple: Dados divididos em treino, validação e teste
        """
        total_samples = len(windows)
        
        # Calcular índices de divisão
        train_end = int(total_samples * train_split)
        val_end = int(total_samples * (train_split + val_split))
        
        # Dividir dados
        train_data = (windows[:train_end], window_labels[:train_end], sensor_ids[:train_end])
        val_data = (windows[train_end:val_end], window_labels[train_end:val_end], sensor_ids[train_end:val_end])
        test_data = (windows[val_end:], window_labels[val_end:], sensor_ids[val_end:])
        
        print(f"Divisão temporal dos dados:")
        print(f"- Treino: {len(train_data[0])} janelas ({len(train_data[0])/total_samples*100:.1f}%)")
        print(f"- Validação: {len(val_data[0])} janelas ({len(val_data[0])/total_samples*100:.1f}%)")
        print(f"- Teste: {len(test_data[0])} janelas ({len(test_data[0])/total_samples*100:.1f}%)")
        
        return train_data, val_data, test_data
    
    def create_tf_dataset(self, windows, labels, shuffle=True):
        """
        Cria dataset TensorFlow
        
        Args:
            windows (np.array): Janelas de dados
            labels (np.array): Labels
            shuffle (bool): Se deve embaralhar
            
        Returns:
            tf.data.Dataset: Dataset TensorFlow
        """
        # Expandir dimensão para (batch_size, window_size, 1)
        windows_expanded = np.expand_dims(windows, axis=-1)
        
        # Criar dataset
        dataset = tf.data.Dataset.from_tensor_slices((windows_expanded, labels))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def build_model(self):
        """
        Constrói o modelo LSTM compartilhado (Estratégia B)
        
        Returns:
            tf.keras.Model: Modelo compilado
        """
        model = keras.Sequential([
            # Camada de entrada
            layers.InputLayer(input_shape=(self.window_size, 1)),
            
            # Primeira camada LSTM
            layers.LSTM(64, return_sequences=True, kernel_initializer=HeNormal()),
            layers.BatchNormalization(),
            
            # Segunda camada LSTM
            layers.LSTM(64, kernel_initializer=HeNormal(), kernel_regularizer=l2(0.01)),
            layers.BatchNormalization(),
            
            # Camadas densas
            layers.Dense(64, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Camada de saída
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compilar modelo
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', Precision(), Recall(), F1Score()]
        )
        
        return model
    
    def train(self, train_data, val_data, epochs=100):
        """
        Treina o modelo
        
        Args:
            train_data (tuple): Dados de treino
            val_data (tuple): Dados de validação
            epochs (int): Número máximo de épocas
            
        Returns:
            tf.keras.callbacks.History: Histórico do treinamento
        """
        train_windows, train_labels, _ = train_data
        val_windows, val_labels, _ = val_data
        
        # Calcular class weights
        self.class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(train_labels), 
            y=train_labels
        )
        class_weight_dict = {i: weight for i, weight in enumerate(self.class_weights)}
        
        print(f"Class weights calculados: {class_weight_dict}")
        
        # Criar datasets
        train_dataset = self.create_tf_dataset(train_windows, train_labels, shuffle=True)
        val_dataset = self.create_tf_dataset(val_windows, val_labels, shuffle=False)
        
        # Construir modelo
        self.model = self.build_model()
        print("Modelo LSTM construído:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_f1_score', 
                mode='max', 
                patience=20, 
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_f1_score', 
                factor=0.5, 
                patience=10, 
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model.keras',
                monitor='val_f1_score',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        # Treinar
        print("Iniciando treinamento...")
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        return history
    
    def evaluate(self, test_data):
        """
        Avalia o modelo no conjunto de teste
        
        Args:
            test_data (tuple): Dados de teste
            
        Returns:
            dict: Métricas de avaliação
        """
        test_windows, test_labels, _ = test_data
        
        print("Avaliando modelo no conjunto de teste...")
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = self.model.evaluate(
            self.create_tf_dataset(test_windows, test_labels, shuffle=False), 
            verbose=0
        )
        
        # Predições detalhadas
        test_predictions = self.model.predict(
            np.expand_dims(test_windows, axis=-1)
        )
        test_predictions_binary = (test_predictions > 0.5).astype(int)
        
        # Relatório de classificação
        print("\nRelatório de Classificação:")
        print(classification_report(test_labels, test_predictions_binary.flatten()))
        
        # Matriz de confusão
        cm = confusion_matrix(test_labels, test_predictions_binary.flatten())
        
        # Plotar matriz de confusão
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Sem Falha', 'Com Falha'], 
                    yticklabels=['Sem Falha', 'Com Falha'])
        plt.title('Matriz de Confusão - Conjunto de Teste')
        plt.ylabel('Valor Real')
        plt.xlabel('Predição')
        plt.show()
        
        return {
            'loss': test_loss,
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'confusion_matrix': cm
        }
    
    def plot_training_history(self, history):
        """
        Plota o histórico de treinamento
        
        Args:
            history (tf.keras.callbacks.History): Histórico do treinamento
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Treino')
        axes[0, 0].plot(history.history['val_loss'], label='Validação')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Treino')
        axes[0, 1].plot(history.history['val_accuracy'], label='Validação')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1-Score
        axes[1, 0].plot(history.history['f1_score'], label='Treino')
        axes[1, 0].plot(history.history['val_f1_score'], label='Validação')
        axes[1, 0].set_title('F1-Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Precision
        axes[1, 1].plot(history.history['precision'], label='Treino')
        axes[1, 1].plot(history.history['val_precision'], label='Validação')
        axes[1, 1].set_title('Precision')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filename='lstm_sensor_fault_detection_final.keras'):
        """
        Salva o modelo treinado
        
        Args:
            filename (str): Nome do arquivo para salvar
        """
        if self.model is not None:
            self.model.save(filename)
            print(f"Modelo salvo como '{filename}'")
        else:
            print("Erro: Modelo não foi treinado ainda")
    
    def save_config(self, filename='model_config.json'):
        """
        Salva as configurações do modelo
        
        Args:
            filename (str): Nome do arquivo para salvar
        """
        config = {
            'window_size': self.window_size,
            'stride': self.stride,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'global_mean': self.global_mean,
            'global_std': self.global_std,
            'class_weights': self.class_weights.tolist() if self.class_weights is not None else None,
            'model_architecture': 'LSTM_Shared_Sensor_Strategy_B',
            'created_at': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configurações salvas como '{filename}'")

def main():
    """
    Função principal para executar o pipeline completo
    """
    print("=" * 80)
    print("DETECÇÃO DE FALHAS EM SENSORES - ESTRATÉGIA B")
    print("=" * 80)
    
    # Inicializar detector
    detector = SensorFaultDetector(
        window_size=240,
        stride=120,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Carregar dados
    df = detector.load_data()
    
    # Pré-processar dados
    sensor_data, binary_labels, sensor_columns = detector.preprocess_data(df)
    
    # Criar janelas por sensor
    windows, window_labels, sensor_ids = detector.create_sensor_windows(
        sensor_data, binary_labels, sensor_columns
    )
    
    # Dividir dados temporalmente
    train_data, val_data, test_data = detector.split_temporal_data(
        windows, window_labels, sensor_ids
    )
    
    # Treinar modelo
    history = detector.train(train_data, val_data, epochs=100)
    
    # Plotar histórico
    detector.plot_training_history(history)
    
    # Avaliar modelo
    results = detector.evaluate(test_data)
    
    # Salvar modelo e configurações
    detector.save_model()
    detector.save_config()
    
    # Relatório final
    print("\n" + "=" * 80)
    print("RELATÓRIO FINAL")
    print("=" * 80)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main() 