print("--- Iniciando prueba de entorno para Python 3.13 ---")

# Comprobamos las librerías básicas de Data Science
try:
    import pandas as pd
    print("\n[OK] Pandas importado correctamente.")
    print(f"   - Versión: {pd.__version__}")
    pd.set_option('display.max_columns', 20)
    print("[OK] pd.set_option() funciona.")
except Exception as e:
    print(f"\n[ERROR] Problema con Pandas: {e}")

try:
    import sklearn
    print("\n[OK] Scikit-learn importado correctamente.")
    print(f"   - Versión: {sklearn.__version__}")
    from sklearn.metrics import classification_report
    print("[OK] import de 'classification_report' funciona.")
except Exception as e:
    print(f"\n[ERROR] Problema con Scikit-learn: {e}")

# Comprobamos TensorFlow (esperamos que falle)
try:
    import tensorflow as tf
    print("\n[INFO] TensorFlow importado correctamente (¡sorprendente!).")
    print(f"   - Versión: {tf.__version__}")
except ImportError:
    print("\n[ADVERTENCIA] TensorFlow no se pudo importar. Esto es lo esperado en Python 3.13.")
except Exception as e:
    print(f"\n[ERROR] Problema inesperado con TensorFlow: {e}")
    
print("\n--- Prueba de entorno finalizada ---")