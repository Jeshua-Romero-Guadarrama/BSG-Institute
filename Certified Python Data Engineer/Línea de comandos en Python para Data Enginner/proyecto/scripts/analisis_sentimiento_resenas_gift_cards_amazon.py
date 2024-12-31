#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
analisis_sentimiento_resenas_gift_cards_amazon.py

Script de línea de comandos que:
1. Descarga recursos de NLTK.
2. Carga datos de reseñas en formato JSONL.
3. Preprocesa textos (normaliza, tokeniza, elimina stopwords, lematiza, etiqueta).
4. Aplica análisis de sentimiento con modelo BERT.
5. Genera y guarda cuatro gráficos en la carpeta 'resultados'.
"""

import os
import sys
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Transformers
from transformers import pipeline
from transformers import BertTokenizer, BertForSequenceClassification

# WordCloud
from wordcloud import WordCloud

# Utilidades
from pathlib import Path
from collections import Counter

def main():
    # =========================================================================
    # 1. DESCARGA DE RECURSOS NLTK
    # =========================================================================
    nltk.download('vader_lexicon')
    nltk.download('words')
    nltk.download('maxent_ne_chunker')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    nltk.download('wordnet')  # Necesario para lematización

    # =========================================================================
    # 2. CARGA DE DATOS
    # =========================================================================
    ruta_jsonl = Path.cwd() / 'data' / 'Gift_Cards_reviews.jsonl'

    try:
        df = pd.read_json(ruta_jsonl, lines=True, encoding='utf-8')
        print(f"[INFO] Datos cargados correctamente desde '{ruta_jsonl}'.")
    except FileNotFoundError:
        print(f"[ERROR] El archivo '{ruta_jsonl}' no se encontró.")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] Error al leer el archivo JSONL: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Ocurrió un error inesperado: {e}")
        sys.exit(1)

    # =========================================================================
    # 3. PREPROCESAMIENTO DE TEXTO
    # =========================================================================
    # 3.1 Función para normalizar el texto
    def normalizar_texto(texto):
        return texto.lower().translate(str.maketrans('', '', string.punctuation))

    df['texto_normalizado'] = df['text'].apply(normalizar_texto)

    # 3.2 Tokenización
    def tokenizar_texto(texto):
        return word_tokenize(texto)

    df['tokens'] = df['texto_normalizado'].apply(tokenizar_texto)

    # 3.3 Eliminación de Stop Words
    english_stopwords = set(stopwords.words('english'))

    def eliminar_stopwords(tokens):
        return [word for word in tokens if word not in english_stopwords]

    df['tokens_sin_stopwords'] = df['tokens'].apply(eliminar_stopwords)

    # 3.4 Lematización
    lematizador = WordNetLemmatizer()

    def lematizar_tokens(tokens):
        return [lematizador.lemmatize(word) for word in tokens]

    df['tokens_lemmatizados'] = df['tokens_sin_stopwords'].apply(lematizar_tokens)

    # 3.5 Etiquetado gramatical
    def etiquetar_gramaticalmente(tokens):
        return nltk.pos_tag(tokens)

    df['etiquetas_gramaticales'] = df['tokens_lemmatizados'].apply(etiquetar_gramaticalmente)

    # =========================================================================
    # 4. ANÁLISIS DE SENTIMIENTOS CON MODELO PREENTRENADO BERT
    # =========================================================================
    def analizar_sentimiento(texto, modelo='nlptown/bert-base-multilingual-uncased-sentiment',
                             max_length=512, device=-1):
        """
        Analiza el sentimiento de un texto utilizando un modelo BERT preentrenado.
        """
        tokenizer = BertTokenizer.from_pretrained(modelo)
        model = BertForSequenceClassification.from_pretrained(modelo)
        clasificador = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
        resultado = clasificador(texto, truncation=True, max_length=max_length)[0]
        return resultado

    # Aplicar análisis de sentimiento
    df['resultado_sentimiento'] = df['text'].apply(analizar_sentimiento)

    # Separar en columnas distintas
    df['etiqueta_sentimiento'] = df['resultado_sentimiento'].apply(lambda x: x['label'])
    df['puntaje_sentimiento'] = df['resultado_sentimiento'].apply(lambda x: x['score'])

    # =========================================================================
    # 5. CLASIFICACIÓN DE SENTIMIENTO EN CATEGORÍAS
    # =========================================================================
    def clasificar_sentimiento(puntaje):
        """
        Clasifica el puntaje de sentimiento en categorías: Insatisfecho, Neutral, Satisfecho.
        """
        if 0 < puntaje <= 0.3:
            return 'Insatisfecho'
        elif 0.3 < puntaje <= 0.5:
            return 'Neutral'
        elif puntaje > 0.5:
            return 'Satisfecho'
        else:
            return 'No definido'

    df['categoria_sentimiento'] = df['puntaje_sentimiento'].apply(clasificar_sentimiento)

    # =========================================================================
    # 6. VISUALIZACIONES
    # =========================================================================
    # Crear carpeta 'resultados' si no existe
    os.makedirs('resultados', exist_ok=True)

    # -------------------------------------------------------------------------
    # 6.1 Nube de Palabras (Gráfico 1)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 4))
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate(' '.join(df['texto_normalizado']))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Nube de Palabras de Reseñas', fontsize=14)

    # Guardar figura
    nombre_figura_1 = 'resultados/Nube de Palabras de Reseñas.png'
    plt.savefig(nombre_figura_1, bbox_inches='tight')
    print(f"[INFO] Se guardó la figura: {nombre_figura_1}")
    plt.close()

    # -------------------------------------------------------------------------
    # 6.2 Top 10 Palabras Más Repetidas (Gráfico 2)
    # -------------------------------------------------------------------------
    palabras_top = pd.DataFrame(
        Counter(palabra for tokens in df['tokens_lemmatizados'] for palabra in tokens).most_common(10),
        columns=['Palabra', 'Frecuencia']
    ).sort_values('Frecuencia')

    plt.figure(figsize=(10, 4))
    sns.barplot(x='Frecuencia', y='Palabra', data=palabras_top, palette='viridis')
    plt.title('Top 10 Palabras Más Repetidas en las Reseñas', fontsize=14)
    plt.xlabel('Frecuencia', fontsize=12)
    plt.ylabel('Palabra', fontsize=12)
    plt.tight_layout()

    nombre_figura_2 = 'resultados/Top 10 Palabras Más Repetidas en las Reseñas.png'
    plt.savefig(nombre_figura_2, bbox_inches='tight')
    print(f"[INFO] Se guardó la figura: {nombre_figura_2}")
    plt.close()

    # -------------------------------------------------------------------------
    # 6.3 Distribución de Calificaciones vs Sentimientos (Gráfico 3)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Distribución de Calificaciones
    sns.countplot(
        x='rating',
        data=df,
        palette='viridis',
        order=sorted(df['rating'].unique()),
        ax=axes[0]
    )
    axes[0].set_title('Distribución de Calificaciones', fontsize=14)
    axes[0].set_xlabel('Calificaciones', fontsize=12)
    axes[0].set_ylabel('Reseñas', fontsize=12)
    axes[0].tick_params(axis='x', rotation=0)

    # Distribución de Sentimientos
    sns.countplot(
        x='categoria_sentimiento',
        data=df,
        palette='viridis',
        order=['Insatisfecho', 'Neutral', 'Satisfecho'],
        ax=axes[1]
    )
    axes[1].set_title('Distribución de Sentimientos', fontsize=14)
    axes[1].set_xlabel('Sentimientos', fontsize=12)
    axes[1].set_ylabel('Reseñas', fontsize=12)
    axes[1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    nombre_figura_3 = 'resultados/Distribución de Calificaciones vs Distribución de Sentimientos.png'
    plt.savefig(nombre_figura_3, bbox_inches='tight')
    print(f"[INFO] Se guardó la figura: {nombre_figura_3}")
    plt.close()

    # -------------------------------------------------------------------------
    # 6.4 Relación entre Calificaciones y Votos Útiles (Gráfico 4)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 4))
    sns.boxplot(x='rating', y='helpful_vote', data=df, palette='viridis')
    plt.title('Relación entre Calificaciones y Votos Útiles', fontsize=14)
    plt.xlabel('Calificación', fontsize=12)
    plt.ylabel('Votos Útiles', fontsize=12)

    nombre_figura_4 = 'resultados/Relación entre Calificaciones y Votos Útiles.png'
    plt.savefig(nombre_figura_4, bbox_inches='tight')
    print(f"[INFO] Se guardó la figura: {nombre_figura_4}")
    plt.close()

    print("[INFO] Proceso completado con éxito.")

if __name__ == "__main__":
    main()