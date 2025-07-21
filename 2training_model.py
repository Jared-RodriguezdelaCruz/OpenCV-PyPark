import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from skimage.transform import resize
from imblearn.over_sampling import RandomOverSampler


def load_and_preprocess_images(folder, label, target_size=(15, 15)):
    images = []
    labels = []
    
    if not os.path.exists(folder):
        print(f"¡Advertencia! Directorio no encontrado: {folder}")
        return np.array([]), np.array([])
    
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img_resized = resize(img, target_size + (3,))
            images.append(img_resized.flatten())
            labels.append(label)
        except Exception as e:
            print(f"Error procesando {img_path}: {str(e)}")
    
    return np.array(images), np.array(labels)

def train_parking_classifier():
    # Verificar existencia de datos
    if not os.path.exists('data/empty') or not os.path.exists('data/occupied'):
        raise FileNotFoundError("Directorios 'data/empty' y 'data/occupied' no encontrados")
    
    empty_count = len(os.listdir('data/empty'))
    occupied_count = len(os.listdir('data/occupied'))
    
    if empty_count == 0 or occupied_count == 0:
        raise ValueError(f"No hay suficientes imágenes. Vacías: {empty_count}, Ocupadas: {occupied_count}")
    
    print(f"\nCargando {empty_count} imágenes vacías y {occupied_count} ocupadas...")
    
    # Cargar datos
    X_empty, y_empty = load_and_preprocess_images('data/empty', 0)
    X_occupied, y_occupied = load_and_preprocess_images('data/occupied', 1)
    
    if len(X_empty) == 0 or len(X_occupied) == 0:
        raise ValueError("No se pudieron cargar imágenes válidas")
    
    X = np.vstack((X_empty, X_occupied))
    y = np.concatenate((y_empty, y_occupied))
    
    # Entrenamiento
    print("\nDividiendo datos...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    print("\nEntrenando modelo...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Evaluación
    print("\nEvaluando modelo...")
    y_pred = model.predict(X_test)
    
    print(f"\nPrecisión: {accuracy_score(y_test, y_pred):.2f}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Guardar modelo
    model_path = 'parking_model.pkl'
    joblib.dump(model, model_path)
    print(f"\nModelo guardado en {model_path}")
    
    return model

if __name__ == "__main__":
    try:
        print("""
        SISTEMA DE ENTRENAMIENTO PARA ESTACIONAMIENTO INTELIGENTE
        ========================================================
        """)
        
        train_parking_classifier()
        
    except Exception as e:
        print(f"\nError durante el entrenamiento: {str(e)}")
        print("\nPosibles soluciones:")
        print("1. Asegúrate de haber capturado imágenes con el script de captura")
        print("2. Verifica que existen los directorios 'data/empty' y 'data/occupied'")
        print("3. Comprueba que hay imágenes en ambos directorios")