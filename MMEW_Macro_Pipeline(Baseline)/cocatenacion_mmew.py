# -*- coding: utf-8 -*-
import json
import numpy as np

ARCHIVOS_ENTRADA = ['Shufle_MMEW_1024d.json', 'Geometric_MMEW_26d.json']
ARCHIVO_SALIDA = 'Combined_MMEW_1050d.json'

def cargar_y_combinar():
    print("\n🔧 Procesando y combinando archivos...")
    with open(ARCHIVOS_ENTRADA[0], 'r') as f1, open(ARCHIVOS_ENTRADA[1], 'r') as f2:
        datos1 = json.load(f1)
        datos2 = json.load(f2)
    
    features1 = np.array(datos1['features'])
    features2 = np.array(datos2['features'])
    labels = np.array(datos1['labels'])
    
    if features1.shape[:2] != features2.shape[:2]:
        raise ValueError("Las dimensiones de frames/características no coinciden")
    
    features_combinadas = np.concatenate([features1, features2], axis=-1)
    
    resultado = {
        'features': features_combinadas.tolist(),
        'labels': labels.tolist(),
        'metadata': {
            'procesamiento': "Combinación horizontal para MMEW Macro",
            'dimensiones_salida': str(features_combinadas.shape)
        }
    }
    
    with open(ARCHIVO_SALIDA, 'w') as f:
        json.dump(resultado, f, indent=2)
    print(f"\n💾 Archivo guardado en {ARCHIVO_SALIDA}")

if __name__ == "__main__":
    cargar_y_combinar()