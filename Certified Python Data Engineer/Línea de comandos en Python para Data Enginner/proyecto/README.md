# Análisis de Sentimiento de Reseñas de Gift Cards de Amazon

## Descripción

Esta aplicación de línea de comandos realiza análisis de sentimientos en reseñas de productos de Gift Cards de Amazon. Procesa los datos, analiza los sentimientos utilizando un modelo BERT preentrenado y genera visualizaciones para interpretar los resultados.

## Estructura del Proyecto

```
proyecto/
│
├── notebooks/
│   ├── analisis_sentimiento_resenas_gift_cards_amazon.ipynb
│   ├── resultados/
│   │   ├── Relación entre Calificaciones y Votos Útiles.png
│   │   ├── Distribución de Calificaciones vs Distribución de Sentimientos.png
│   │   ├── Top 10 Palabras Más Repetidas en las Reseñas.png
│   │   └── Nube de Palabras de Reseñas.png
│   └── data/
│       └── Gift_Cards_reviews.jsonl
│
├── scripts/
│   ├── analisis_sentimiento_resenas_gift_cards_amazon.py
│   ├── resultados/
│   └── data/
│       └── Gift_Cards_reviews.jsonl
│
├── run.sh
├── environment.yml
├── requirements.txt
└── README.md
```

## Requisitos

- Anaconda (recomendado)
- Python 3.7 o superior

## Instalación y Ejecución

1. **Clonar el Repositorio:**

   ```bash
   git clone https://github.com/tu_usuario/sentiment_analysis_project.git
   cd sentiment_analysis_project
   ```

2. **Colocar el Archivo de Datos:**

   Asegúrate de que el archivo Gift_Cards_reviews.jsonl esté en las carpetas notebooks/data/ y scripts/data/.

3. **Configurar el Entorno y Ejecutar el Análisis:**

   Si usas Anaconda:

   ```bash
   ./run.sh
   ```

   Si no usas Anaconda, crea un entorno virtual y instala las dependencias:

   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python analisis_sentimiento_resenas_gift_cards_amazon.py
   ```

## Resultados

- **Gráficos Generados:**
  - `Relación entre Calificaciones y Votos Útiles.png`
  - `Distribución de Calificaciones vs Distribución de Sentimientos.png`
  - `Top 10 Palabras Más Repetidas en las Reseñas.png`
  - `Nube de Palabras de Reseñas.png`

- **Datos Procesados:**
  - `processed_reviews.csv` en `scripts/data/`

## Notas

- Asegúrate de tener una conexión a internet estable para descargar los modelos preentrenados y recursos de NLTK.
- Para utilizar GPU en el análisis de sentimientos, ajusta el parámetro `device` en la función `analizar_sentimiento` dentro del script `analisis_sentimiento_resenas_gift_cards_amazon.py`.