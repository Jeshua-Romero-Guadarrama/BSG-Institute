#!/bin/bash

# run.sh
# Script para configurar el entorno y ejecutar el análisis de sentimiento

echo "Iniciando el script de ejecución para el Análisis de Sentimiento..."

# Crear el entorno de Anaconda
echo "Creando el entorno de Anaconda..."
conda env create -f environment.yml

# Activar el entorno
echo "Activando el entorno de Anaconda..."
source activate sentiment_env

# Actualizar pip
echo "Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias adicionales si es necesario
pip install -r requirements.txt

# Ejecutar el script de análisis
echo "Ejecutando el script de análisis de sentimiento..."
python scripts/analisis_sentimiento_resenas_gift_cards_amazon.py

echo "Proceso completado exitosamente."
