# InmoSmart - Prototipo local

InmoSmart integra los resultados de los tres talleres del proyecto:

- Aprendizaje supervisado: estimación del valor justo.
- Aprendizaje no supervisado: segmentación del mercado con K-Means.
- Aprendizaje por refuerzo: recomendación de comprar, negociar o seguir buscando.

## Requisitos

- Python 3 instalado.
- Navegador moderno.
- Conexión a Internet.

## Ejecutar localmente

Desde una terminal de Windows:

```powershell
python -m pip install -r requirements.txt
python -m modelos.entrenar_todos
python app.py
```

Abrir en el navegador:

```text
http://127.0.0.1:8000/
```

No abrir `index.html` directamente con doble clic. El servidor Flask publica la página y las APIs.

Para detener el servidor, presionar `Ctrl+C` en la terminal.

## Archivos principales

- `Pagina/index.html`: estructura, navegación y formulario.
- `Pagina/script.js`: interfaz y consumo de las APIs.
- `Pagina/styles.css`: interfaz visual.
- `app.py`: servidor Flask y integración de los modelos.
- `modelos/entrenar_supervisado.py`: LightGBM o Random Forest.
- `modelos/entrenar_kmeans.py`: aprendizaje no supervisado.
- `modelos/entrenar_refuerzo.py`: Q-Learning.
- `modelos/entrenar_todos.py`: ejecuta todo el entrenamiento.
- `dataset_limpio.csv`: 8.372 registros utilizados por el prototipo.

## Aprendizaje supervisado

La estimación supervisada se implementa en `predictSupervised()` en `script.js`.

El sistema:

1. Filtra el dataset a las propiedades con área, precio y estrato válidos.
2. Busca los 40 anuncios más cercanos al inmueble ingresado.
3. Compara área construida, estrato, habitaciones, baños, parqueaderos y administración.
4. Asigna mayor peso a los anuncios más similares.
5. Combina el precio comparable ponderado con:
   - 65 %: mediana ponderada de comparables.
   - 25 %: mediana de la ubicación seleccionada.
   - 10 %: mediana del estrato.

Esta implementación utiliza los datos reales del proyecto y evita la fórmula estática de simulación usada originalmente. El prototipo no carga directamente LightGBM dentro del navegador; representa el resultado supervisado mediante vecinos comparables y medianas, conservando la lógica de valuación.

## Aprendizaje no supervisado

El clustering se implementa en `kmeansProperties()` en `script.js`.

Se normalizan y agrupan tres variables:

- Área construida.
- Estrato.
- Precio.

La implementación calcula tres centroides K-Means y los ordena por precio. Se muestran como Económico, Medio y Premium. Para un inmueble nuevo, `assignCluster()` asigna:

- Premium: área mayor a 200 m² y estrato mayor o igual a 5.
- Económico: área menor a 100 m² y estrato menor o igual a 4.
- Medio: los demás casos.

El dashboard representa cada centroide como una burbuja mediante el área construida y el estrato promedio.

## Semáforo de precio

`calculatePrice()` compara el precio publicado con el valor predicho:

- Oportunidad: el precio publicado es al menos 15 % menor que el valor predicho.
- Sobrevalorado: el precio publicado es al menos 15 % mayor que el valor predicho.
- Precio justo: la diferencia está dentro de ±15 %.

El umbral ±15 % coincide con la definición utilizada en el notebook MDP.

## Agente de refuerzo

La recomendación del agente se implementa en `getRLPolicy()` en `script.js`.

El MDP utiliza nueve estados posibles:

```text
Segmento × Posición de precio
Económico / Medio / Premium
×
Oportunidad / Justo / Sobrevalorado
```

Las acciones disponibles son:

- Comprar.
- Negociar fuerte.
- Negociar suave.
- Seguir buscando.

La política aplicada al prototipo representa las decisiones aprendidas en los notebooks:

- Oportunidad: comprar.
- Sobrevalorado económico: seguir buscando.
- Sobrevalorado en otros segmentos: negociar fuerte.
- Justo premium: negociar fuerte.
- Justo económico o medio: negociar fuerte como política de mejora de ahorro.

El notebook original aplica Q-Learning ε-greedy, recompensas basadas en ahorro y comparación con Value Iteration. En el navegador se muestra la política obtenida de forma auditable, sin descargar el modelo PyTorch ni el entorno completo.

## Flujo funcional

1. `loadDataset()` carga `dataset_limpio.csv`.
2. Se completa el catálogo de ubicaciones.
3. `calculatePrice()` obtiene los datos del formulario.
4. `predictSupervised()` estima el valor justo.
5. `assignCluster()` asigna el perfil de mercado.
6. Se calcula la brecha entre precio publicado y valor justo.
7. `getRLPolicy()` selecciona la acción del agente.
8. `renderPriceChart()` muestra la comparación gráfica.
9. `renderClusterChart()` muestra los segmentos de mercado.

## Datos cargados

El prototipo excluye de forma efectiva las variables derivadas del precio. En particular no utiliza `precio_m2_calc`, evitando una fuga de datos. El dataset final contiene 8.372 registros y 820 ubicaciones normalizadas.
