# Pipeline integral basado en modelos de lenguaje de gran tamaño para la detección del riesgo de suicidio y la caracterización de perfiles vulnerables en redes sociales desde un enfoque de IA responsable

**Autores:** Laura Pedro-Botet · Anna Pujadas · Walther Jesus Nuñez

Desarrollado en el marco del proyecto **STOP** (*Suicide prevenTion in sOcial Platforms*).

---

## Descripción general

Este repositorio contiene la implementación completa de un pipeline integral y preservador de privacidad basado en Modelos de Lenguaje de Gran Tamaño (LLMs) para la detección y caracterización de usuarios de redes sociales que pueden presentar indicadores de ideación suicida, con cumplimiento del Reglamento General de Protección de Datos (RGPD).

El enfoque no aborda la detección del riesgo suicida como una tarea de clasificación aislada, sino que integra múltiples tareas de inferencia basadas en LLMs dentro de un flujo de trabajo unificado: desde la recopilación y anonimización de datos hasta la evaluación del riesgo, la validación por expertos y la caracterización de la población objetivo para el diseño de una campaña piloto de prevención.

### Resultados principales

| Etapa | Métrica | Valor |
|---|---|---|
| Anonimización (Arquitectura B) | Recall | 0,93 |
| Anonimización (Arquitectura B) | Precisión | 0,81 |
| Cribado de salud mental | Publicaciones positivas | 2,24% de 103.334 |
| Riesgo suicida - Usuarios Positivos | - | 11,59% (72/621) |
| Riesgo suicida - Usuarios Dudosos | - | 53,46% (332/621) |
| Acuerdo expertos con el pipeline | - | 68,56% (277/404) |
| Usuarios positivos validados | Género (femenino) | 58% |
| Usuarios positivos validados | Edad 18–34 | 78% |

---

## Arquitectura del pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│  1. Recopilación de datos     Publicaciones X (2020–2025, ESP) · keywords│
│  2. LLM_UserType              Individual vs. Organización  [Gemma 3 12B] │
│  3. Descarga de historial     Hasta últimas 100 publicaciones            │
│  4. Anonimización             Regex → Presidio+BERT → LLaMA 3 8B         │
│     + Generalización bio      profile_bio → semántica → ES→FR            │
│  5. LLM_MH                    Cribado salud mental  Positivo / Control   │
│  6. LLM_SuicideRisk           Riesgo suicida  Positivo / Dudoso / Neg.   │
│  7. Validación por expertos   Revisión por profesionales de salud mental │
│  8. LLM_Interests             Categorización de intereses (198 temas)    │
│  9. LLM_Demographics          Inferencia de género y rango de edad       │
│ 10. Campaña (futuro)          Campaña piloto en TikTok (15 días)         │
└──────────────────────────────────────────────────────────────────────────┘
```

### Comparativa de arquitecturas de anonimización

| Arquitectura | Motor NER | Verificación LLM | Recall | Precisión |
|---|---|---|---|---|
| A | Presidio + spaCy | LLaMA 3 8B | 0,91 | 0,43 |
| **B (seleccionada)** | **Presidio + BERT** | **LLaMA 3 8B** | **0,93** | **0,81** |
| C | - | Gemma 3 12B (directo) | 0,92 | 0,62 |

---

## Estructura del repositorio

| Notebook / Archivo | Etapa del pipeline | Descripción |
|---|---|---|
| `Pipeline.ipynb` | Etapas 1–4 | **Pipeline unificado principal.** Recopilación de datos, filtrado por tipo de usuario, descarga del historial, anonimización multicapa (Regex → Presidio+BERT → LLaMA 3 8B) y generalización semántica de `profile_bio`. Cumplimiento RGPD: los datos personales crudos nunca se escriben en disco. |
| `LLM_SM.py` | Etapa 5 | Script de cribado de salud mental. Clasifica cada publicación anonimizada como **Positivo** (contenido de salud mental en primera persona) o **Control**. Utiliza Gemini Flash Lite. |
| `LLM_SuicideRisk.ipynb` | Etapa 6 | Clasificación del riesgo suicida a nivel de usuario. Concatena cronológicamente las publicaciones positivas de cada usuario y asigna **Positivo / Dudoso / Negativo**. Utiliza Gemini Flash Lite. |
| `Agrupacion_usuarios.ipynb` | Etapa 7 | Construcción del dataset a nivel de usuario. |
| `Eval_LLM_SuicideRisk.ipynb` | Etapa 8 | Análisis de validación por expertos. Calcula métricas de acuerdo y matriz de confusión entre las predicciones automáticas y las anotaciones de profesionales de salud mental. |
| `LLM_Interests.ipynb` | Etapa 9 | Categorización de intereses. Asigna hasta 10 etiquetas de interés por usuario positivo validado a partir de una taxonomía de 198 temas usando Gemini Flash Lite. |
| `EDA_LLM_Interests.ipynb` | Etapa 10 | Análisis exploratorio de los resultados de categorización de intereses. Visualizaciones de las principales categorías de interés en la población de usuarios validados. |
| `LLM_Demographics.ipynb` | Etapa 11 | Inferencia demográfica. Estima el género y el rango de edad por usuario a partir de publicaciones anonimizadas y `profile_bio` generalizada. Utiliza Gemini Flash Lite. |

---

## Modelos utilizados

| Modelo | Rol | Despliegue |
|---|---|---|
| **Gemma 3 12B Instruct** (`unsloth/gemma-3-12b-it`) | Clasificación tipo de usuario · Generalización bio | Local (HuggingFace, cuantización 4-bit NF4) |
| **LLaMA 3 8B Instruct** (`llama3`) | Detección residual de PII (*Personally Identifiable Information*) (Etapa 4c) | Local (Ollama, sin token HuggingFace) |
| **BERT multilingüe NER** (`Davlan/bert-base-multilingual-cased-ner-hrl`) | Reconocimiento de entidades nombradas (Presidio) | Local (HuggingFace) |
| **Helsinki-NLP MarianMT** | Normalización de idioma + traducción ES→FR | Local (HuggingFace) |
| **Gemini 2.0 Flash Lite** | Cribado salud mental · Riesgo suicida · Intereses · Demografía | Nube (Google AI API) |

> **Nota de privacidad:** Gemma 3 12B y LLaMA 3 8B se ejecutan localmente para garantizar que ningún dato personal no anonimizado sea transmitido a servicios externos. Gemini en la nube solo se invoca una vez completada la anonimización.

---

## Requisitos

### Hardware
- **Mínimo:** GPU NVIDIA T4 (15 GB VRAM) - requiere `SWAP_MODELS=True` en `Pipeline.ipynb`
- **Recomendado:** GPU NVIDIA A100 (40 GB VRAM) - establecer `SWAP_MODELS=False`
- Se recomienda Google Colab (Pro o Pro+) para ejecutar `Pipeline.ipynb`

### APIs necesarias
- Clave API de [twitterapi.io](https://twitterapi.io) - para la recopilación de tweets (Etapas 1 y 3).
- Clave API de Google Gemini - para las Etapas 5, 6, 9 y 11.

### Dependencias Python (instaladas automáticamente en la Celda 1 de `Pipeline.ipynb`)

```bash
pip install transformers>=4.50.0 accelerate bitsandbytes
pip install presidio-analyzer[transformers] presidio-anonymizer
pip install spacy && python -m spacy download es_core_news_sm
pip install langdetect sentencepiece sacremoses tqdm requests ollama
apt-get install zstd
curl -fsSL https://ollama.com/install.sh | sh
```

---

## Uso

### Ejecución del pipeline completo (`Pipeline.ipynb`)

1. Abrir `Pipeline.ipynb` en Google Colab con runtime de GPU habilitado.
2. **Celda 1** - Instalar todas las dependencias. Tras la ejecución, hacer **Runtime → Reiniciar sesión**.
3. **Celda 2** - Configurar los parámetros:
   ```python
   TWITTER_API_KEY = "tu_clave_aquí"
   DEBUG_MODE = False # True = modo prueba (3 fechas, 4 keywords, 2 usuarios)
   MAX_USERS_DEBUG = None # None = todos los usuarios
   SWAP_MODELS = True # True para T4; False para A100
   ```
4. **Celda 3** - Definición de funciones (ejecutar sin modificar).
5. **Celda 4** - Montar Google Drive y cargar todos los modelos (Gemma, Presidio+BERT, traductores Helsinki, Ollama/LLaMA).
6. **Celda 5** - Ejecutar `run_pipeline()`. Resultado: DataFrame `df_final` (anonimizado, conforme al RGPD).

### Modo debug / prueba rápida

Para validar el pipeline de extremo a extremo sin consumir créditos de API:

```python
# Celda 2
DEBUG_MODE = True
MAX_USERS_DEBUG = 2
DEBUG_SPECIFIC_DATES = ["2023-03-28", "2022-11-22", "2020-12-23"]
DEBUG_KEYWORDS = ["tengo ansiedad", "no puedo parar de llorar",
                        "sin ganas de vivir", "Suicida"]
```

### Reanudación de sesión interrumpida

```python
SESSION_OFFSET = 150   # omite los primeros 150 usuarios ya procesados
```

Un archivo de checkpoint en formato JSONL almacenado en Google Drive (`CHECKPOINT_PATH`) guarda los registros anonimizados tras cada usuario, permitiendo reanudar el proceso entre sesiones.

---

## Diseño de privacidad (RGPD)

- **Los datos personales crudos nunca se escriben en disco.** Todo el procesamiento intermedio ocurre en memoria RAM; solo el output ya anonimizado puede persistirse.
- **Salt de sesión volátil.** Los IDs de usuario y de tweet se hashean con SHA-256 utilizando un salt aleatorio no persistente generado por sesión (`secrets.token_hex(32)`), haciendo imposible la re-identificación entre sesiones.
- **LLMs locales para las etapas previas a la anonimización.** Gemma 3 12B y LLaMA 3 8B se ejecutan localmente; ningún dato personal sin anonimizar llega a APIs externas.
- **Generalización semántica de biografías.** Los campos `profile_bio` se abstraen en descripciones demográficas genéricas antes de cualquier procesamiento en la nube.

---

## Cita

Si utilizas este código o metodología en tu investigación, por favor cita:

```
Pedro-Botet, L., Pujadas, A., & Nuñez, W. J. (2026).
Pipeline integral basado en modelos de lenguaje de gran tamaño para la detección
del riesgo de suicidio y la caracterización de perfiles vulnerables en redes sociales
desde un enfoque de IA responsable.
```

---

## Declaración ética

Esta investigación se realizó utilizando datos públicos de redes sociales. Todo el procesamiento cumple con los requisitos del RGPD. El pipeline está diseñado para **apoyar**, no reemplazar, el juicio clínico profesional. Ningún usuario clasificado como Positivo por el sistema automático fue posteriormente evaluado como Negativo por los expertos en salud mental, lo que demuestra que el sistema prioriza la seguridad minimizando los falsos negativos. La recopilación y el análisis de datos se realizaron en colaboración con especialistas en salud mental.
