#!/usr/bin/env python3
"""
Ejemplo de uso del Sensor Fault Detector
Estratégia B: LSTM Compartilhada "um sensor por vez"

Este script demonstra como usar a classe SensorFaultDetector para treinar
e avaliar um modelo de detecção de falhas em sensores.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sensor_fault_detector import SensorFaultDetector

def example_basic_usage():
    """
    Exemplo básico de uso do detector
    """
    print("=" * 60)
    print("EXEMPLO BÁSICO - DETECÇÃO DE FALHAS EM SENSORES")
    print("=" * 60)
    
    # 1. Inicializar o detector com parâmetros padrão
    detector = SensorFaultDetector(
        window_size=240,    # Janela temporal
        stride=120,         # Passo entre janelas (50% overlap)
        batch_size=32,      # Tamanho do batch
        learning_rate=0.001 # Taxa de aprendizagem
    )
    
    print("✓ Detector inicializado")
    
    # 2. Carregar dados
    try:
        df = detector.load_data(data_dir='dataset/')
        print("✓ Dados carregados com sucesso")
    except Exception as e:
        print(f"✗ Erro ao carregar dados: {e}")
        return
    
    # 3. Pré-processar dados
    sensor_data, binary_labels, sensor_columns = detector.preprocess_data(df)
    print("✓ Dados pré-processados")
    
    # 4. Criar janelas por sensor (Estratégia B)
    windows, window_labels, sensor_ids = detector.create_sensor_windows(
        sensor_data, binary_labels, sensor_columns
    )
    print("✓ Janelas temporais criadas")
    
    # 5. Dividir dados temporalmente
    train_data, val_data, test_data = detector.split_temporal_data(
        windows, window_labels, sensor_ids
    )
    print("✓ Dados divididos temporalmente")
    
    # 6. Treinar modelo
    print("\nIniciando treinamento...")
    history = detector.train(train_data, val_data, epochs=50)  # Reduzido para exemplo
    print("✓ Modelo treinado")
    
    # 7. Avaliar modelo
    print("\nAvaliando modelo...")
    results = detector.evaluate(test_data)
    print("✓ Modelo avaliado")
    
    # 8. Salvar modelo e configurações
    detector.save_model('exemplo_modelo.keras')
    detector.save_config('exemplo_config.json')
    print("✓ Modelo e configurações salvos")
    
    # 9. Mostrar resultados
    print("\n" + "=" * 40)
    print("RESULTADOS FINAIS")
    print("=" * 40)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1_score']:.4f}")
    print("=" * 40)

def example_parameter_tuning():
    """
    Exemplo de ajuste de parâmetros
    """
    print("\n" + "=" * 60)
    print("EXEMPLO DE AJUSTE DE PARÂMETROS")
    print("=" * 60)
    
    # Diferentes configurações para testar
    configurations = [
        {
            'name': 'Configuração 1 - Janela Pequena',
            'window_size': 120,
            'stride': 60,
            'batch_size': 32,
            'learning_rate': 0.001
        },
        {
            'name': 'Configuração 2 - Janela Grande',
            'window_size': 480,
            'stride': 240,
            'batch_size': 32,
            'learning_rate': 0.001
        },
        {
            'name': 'Configuração 3 - Learning Rate Menor',
            'window_size': 240,
            'stride': 120,
            'batch_size': 32,
            'learning_rate': 0.0001
        }
    ]
    
    results_comparison = []
    
    for config in configurations:
        print(f"\nTestando: {config['name']}")
        
        # Criar detector com configuração específica
        detector = SensorFaultDetector(
            window_size=config['window_size'],
            stride=config['stride'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate']
        )
        
        try:
            # Carregar e processar dados
            df = detector.load_data(data_dir='dataset/')
            sensor_data, binary_labels, sensor_columns = detector.preprocess_data(df)
            windows, window_labels, sensor_ids = detector.create_sensor_windows(
                sensor_data, binary_labels, sensor_columns
            )
            train_data, val_data, test_data = detector.split_temporal_data(
                windows, window_labels, sensor_ids
            )
            
            # Treinar com menos épocas para comparação rápida
            history = detector.train(train_data, val_data, epochs=20)
            
            # Avaliar
            results = detector.evaluate(test_data)
            
            # Guardar resultados
            results_comparison.append({
                'config_name': config['name'],
                'f1_score': results['f1_score'],
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall']
            })
            
            print(f"✓ {config['name']}: F1-Score = {results['f1_score']:.4f}")
            
        except Exception as e:
            print(f"✗ Erro na {config['name']}: {e}")
    
    # Mostrar comparação
    print("\n" + "=" * 60)
    print("COMPARAÇÃO DE CONFIGURAÇÕES")
    print("=" * 60)
    
    for result in results_comparison:
        print(f"\n{result['config_name']}:")
        print(f"  F1-Score:  {result['f1_score']:.4f}")
        print(f"  Accuracy:  {result['accuracy']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall:    {result['recall']:.4f}")

def example_analysis_per_sensor():
    """
    Exemplo de análise detalhada por sensor
    """
    print("\n" + "=" * 60)
    print("EXEMPLO DE ANÁLISE POR SENSOR")
    print("=" * 60)
    
    # Inicializar detector
    detector = SensorFaultDetector()
    
    try:
        # Carregar e processar dados
        df = detector.load_data(data_dir='dataset/')
        sensor_data, binary_labels, sensor_columns = detector.preprocess_data(df)
        windows, window_labels, sensor_ids = detector.create_sensor_windows(
            sensor_data, binary_labels, sensor_columns
        )
        train_data, val_data, test_data = detector.split_temporal_data(
            windows, window_labels, sensor_ids
        )
        
        # Treinar modelo
        history = detector.train(train_data, val_data, epochs=30)
        
        # Análise por sensor
        test_windows, test_labels, test_sensor_ids = test_data
        unique_sensors = np.unique(test_sensor_ids)
        
        print(f"Analisando desempenho de {len(unique_sensors)} sensores...")
        
        sensor_performance = []
        
        for sensor_id in unique_sensors:
            # Filtrar dados do sensor
            sensor_mask = test_sensor_ids == sensor_id
            sensor_windows = test_windows[sensor_mask]
            sensor_labels = test_labels[sensor_mask]
            
            if len(sensor_windows) == 0:
                continue
            
            # Predições
            sensor_windows_expanded = np.expand_dims(sensor_windows, axis=-1)
            sensor_predictions = detector.model.predict(sensor_windows_expanded, verbose=0)
            sensor_predictions_binary = (sensor_predictions > 0.5).astype(int).flatten()
            
            # Métricas
            accuracy = np.mean(sensor_predictions_binary == sensor_labels)
            
            # Precision, Recall, F1
            tp = np.sum((sensor_predictions_binary == 1) & (sensor_labels == 1))
            fp = np.sum((sensor_predictions_binary == 1) & (sensor_labels == 0))
            fn = np.sum((sensor_predictions_binary == 0) & (sensor_labels == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            sensor_performance.append({
                'sensor_id': sensor_id,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'samples': len(sensor_windows),
                'failures': np.sum(sensor_labels == 1)
            })
        
        # Mostrar top 10 sensores
        sensor_df = pd.DataFrame(sensor_performance)
        sensor_df = sensor_df.sort_values('f1', ascending=False)
        
        print("\nTop 10 sensores por F1-Score:")
        print(sensor_df.head(10)[['sensor_id', 'f1', 'accuracy', 'precision', 'recall', 'samples', 'failures']])
        
        # Plotar distribuição de F1-Score por sensor
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(sensor_df['f1'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribuição de F1-Score por Sensor')
        plt.xlabel('F1-Score')
        plt.ylabel('Número de Sensores')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.scatter(sensor_df['samples'], sensor_df['f1'], alpha=0.6, color='red')
        plt.title('F1-Score vs Número de Amostras')
        plt.xlabel('Número de Amostras')
        plt.ylabel('F1-Score')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Erro na análise por sensor: {e}")

def main():
    """
    Função principal que executa todos os exemplos
    """
    print("EXEMPLOS DE USO - DETECTOR DE FALHAS EM SENSORES")
    print("Estratégia B: LSTM Compartilhada 'um sensor por vez'")
    print("\nEste script demonstra diferentes aspectos do detector:")
    print("1. Uso básico completo")
    print("2. Ajuste de parâmetros")
    print("3. Análise por sensor")
    
    # Executar exemplos
    example_basic_usage()
    example_parameter_tuning()
    example_analysis_per_sensor()
    
    print("\n" + "=" * 60)
    print("TODOS OS EXEMPLOS CONCLUÍDOS!")
    print("=" * 60)
    print("\nPara mais informações, consulte:")
    print("- README.md: Documentação completa")
    print("- LSTM_Sensor_Fault_Detection.ipynb: Notebook interativo")
    print("- sensor_fault_detector.py: Implementação principal")

if __name__ == "__main__":
    main() 