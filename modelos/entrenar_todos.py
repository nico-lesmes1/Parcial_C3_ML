from . import entrenar_supervisado
from . import entrenar_kmeans
from . import entrenar_refuerzo


def main():
    print("\n=== 1. Entrenamiento supervisado ===")
    entrenar_supervisado.main()
    print("\n=== 2. Entrenamiento no supervisado ===")
    entrenar_kmeans.main()
    print("\n=== 3. Entrenamiento por refuerzo ===")
    entrenar_refuerzo.main()
    print("\nTodos los artefactos quedaron en la carpeta artifacts/.")


if __name__ == "__main__":
    main()
