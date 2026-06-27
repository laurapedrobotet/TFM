# LLM_SM - Clasificación de riesgo en salud mental

import os
import json
import time
import random
import logging
import re 

import pandas as pd

from google import genai
from google.genai import types

from datetime import datetime, timedelta

# ==========================================
# CONFIG
# ==========================================

API_KEY = ""

client = genai.Client(api_key=API_KEY)

MODELS = [
    "gemini-3.1-flash-lite"
]

BATCH_SIZE = 30 # Número de tweets procesador por llamada al modelo
SAVE_EVERY = 20 # Cada cuántos batches se guarda checkpoint

# ==========================================
# PATHS
# ==========================================

TFM_PATH = r"C:/Users/Prestec/Desktop/TFM"

INPUT_PATH = os.path.join(
    TFM_PATH,
    "suicidio_anonymized.csv"
)

OUTPUT_DIR = os.path.join(
    TFM_PATH,
    "positivo_control"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "tweets_clasificados_mental_health.csv"
)

OUTPUT_PATH_JSON = os.path.join(
    OUTPUT_DIR,
    "tweets_clasificados_mental_health.json"
)

FINAL_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "mental_health_anonymized_clasif.csv"
)

LOG_PATH = os.path.join(
    OUTPUT_DIR,
    "classification.log"
)

# ==========================================
# LOGGING
# ==========================================

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# ==========================================
# LOAD DATA
# ==========================================

print("Cargando dataset...")

df_original = pd.read_csv(INPUT_PATH)

df = df_original[["id", "text"]].copy()

# ==========================================
# RESUME PREVIOUS EXECUTION
# ==========================================

if os.path.exists(OUTPUT_PATH):

    print("\n📂 Checkpoint encontrado")

    previous_results = pd.read_csv(OUTPUT_PATH)

    processed_ids = set(previous_results["id"])

    print(f"Tweets ya procesados: {len(processed_ids)}")

    # Filtrar tweets pendientes

    df = df[~df["id"].isin(processed_ids)].copy()

    print(f"Tweets pendientes: {len(df)}")

else:

    previous_results = pd.DataFrame()

print(f"Tweets cargados: {len(df)}")

# ==========================================
# PROMPT BASE
# ==========================================

SYSTEM_PROMPT = """
Eres un psicólogo experto en salud mental y análisis de riesgo psicológico en redes sociales.

Clasifica cada tweet en UNA categoría:

- P:
El usuario expresa riesgo psicológico:
depresión, ideación suicida, desesperanza,
autolesión, TCA, ansiedad severa,
soledad extrema o sufrimiento emocional grave.

- C:
El tweet NO expresa riesgo psicológico.
Incluye:
noticias, bromas, opiniones,
concienciación, apoyo a terceros,
conversaciones generales.

IMPORTANTE:
Hablar sobre salud mental NO implica riesgo psicológico.
El riesgo debe ser expresado por el autor del tweet.

Devuelve SOLO un JSON válido.

Formato:
[
  {
    "id": "123",
    "clasif": "P"
  }
]
"""

# ==========================================
# BUILD BATCH PROMPT
# ==========================================

def build_batch_prompt(batch_df):

    tweets_text = ""

    for _, row in batch_df.iterrows():

        tweet_id = row["id"]
        tweet_text = str(row["text"]).strip()[:500]

        tweets_text += f"""
ID: {tweet_id}
Tweet: {tweet_text}

"""

    return SYSTEM_PROMPT + "\n\nTweets:\n" + tweets_text

# ==========================================
# WAIT UNTIL NEXT DAY
# ==========================================

def wait_until_next_day():

    now = datetime.now()

    tomorrow = now + timedelta(days=1)

    next_reset = datetime(
        year=tomorrow.year,
        month=tomorrow.month,
        day=tomorrow.day,
        hour=0,
        minute=5
    )

    seconds_to_wait = max(
    (next_reset - now).total_seconds(),
    0
    )

    hours = seconds_to_wait / 3600

    print(
        f"\n⏳ Esperando hasta mañana "
        f"({hours:.2f} horas)"
    )

    logging.warning(
        f"Esperando {hours:.2f} horas "
        f"hasta reset de cuota"
    )

    time.sleep(seconds_to_wait)

# ==========================================
# API CALL
# ==========================================

def classify_batch(prompt, max_retries=3):

    for model_name in MODELS:

        attempt = 0

        while attempt < max_retries:

            try:

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                        max_output_tokens=2000
                    )
                )

                response_text = response.text

                # Empty response
                if response_text is None:

                    raise ValueError(
                        "Respuesta vacía del modelo"
                    )

                response_text = response_text.strip()

                if response_text == "":

                    raise ValueError(
                        "Respuesta vacía del modelo"
                    )

                # Direct json parse
                try:

                    return json.loads(response_text)

                # Recover json with regex
                except:

                    match = re.search(
                        r"\[.*\]",
                        response_text,
                        re.DOTALL
                    )

                    if match:

                        return json.loads(
                            match.group()
                        )

                    raise ValueError("JSON inválido")

            except Exception as e:

                error_msg = str(e)

                print(f"\n⚠️ Error con {model_name}")
                print(error_msg[:300])

                logging.error(
                    f"Modelo: {model_name} | "
                    f"Error: {error_msg}"
                )

                # Resource exhausted
                if "RESOURCE_EXHAUSTED" in error_msg:

                    print(
                        "\n⚠️ RESOURCE_EXHAUSTED detectado"
                    )

                    # Daily quota
                    if (
                        "GenerateRequestsPerDay"
                        in error_msg
                        or "PerDay" in error_msg
                    ):

                        print(
                            "\n⛔ Cuota diaria agotada"
                        )

                        logging.warning(
                            "Cuota diaria agotada"
                        )

                        wait_until_next_day()

                        # Reiniciar retries
                        attempt = 0

                        continue

                    # Temporary limit
                    else:

                        wait = 300  # 5 min

                        print(
                            f"\n⏳ Saturación temporal. "
                            f"Esperando "
                            f"{wait/60:.0f} min"
                        )

                        logging.warning(
                            "RESOURCE_EXHAUSTED temporal"
                        )

                        time.sleep(wait)

                        # Reiniciar retries
                        attempt = 0

                        continue

                # Normal retry
                wait = max(
                    (2 ** attempt)
                    + random.uniform(0, 1),
                    15
                )

                print(
                    f"Reintentando "
                    f"en {wait:.1f}s"
                )

                time.sleep(wait)

                attempt += 1

    return None

# ==========================================
# SAVE CHECKPOINT
# ==========================================

def save_checkpoint(results):

    temp_df = pd.DataFrame(results)

    temp_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    temp_df.to_json(
        OUTPUT_PATH_JSON,
        orient="records",
        force_ascii=False
    )

# ==========================================
# MAIN
# ==========================================

def main():

    results = previous_results.to_dict("records")

    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Total batches: {total_batches}")

    logging.info(f"Inicio clasificación | batches={total_batches}")

    try:

        for batch_num in range(total_batches):

            start = batch_num * BATCH_SIZE
            end = start + BATCH_SIZE

            batch_df = df.iloc[start:end]

            # Progress log
            progress_msg = (
                f"Batch {batch_num+1}/{total_batches} | "
                f"Tweets procesados: {min(end, len(df))}/{len(df)}"
            )

            print(f"\n{progress_msg}")

            logging.info(progress_msg)

            # Build prompt
            prompt = build_batch_prompt(batch_df)

            # Classify
            parsed = classify_batch(prompt)

            # Handle failures
            if parsed is None:

                print("Batch fallido.")

                logging.error(f"Batch fallido: {batch_num+1}")

                for _, row in batch_df.iterrows():

                    results.append({
                        "id": row["id"],
                        "clasif": "ERROR"
                    })

                continue

            # Track returned ids
            returned_ids = set()

            # Save results
            for item in parsed:

                tweet_id = item.get("id")

                returned_ids.add(tweet_id)

                results.append({
                    "id": tweet_id,
                    "clasif": item.get("clasif", "ERROR")
                })

            # CHECK MISSING IDS
            batch_ids = set(batch_df["id"])

            missing_ids = batch_ids - returned_ids

            for missing_id in missing_ids:

                results.append({
                    "id": missing_id,
                    "clasif": "ERROR"
                })

            # Periodic save
            if (batch_num + 1) % SAVE_EVERY == 0:

                save_checkpoint(results)

                print(
                    f"\n💾 Guardado parcial: "
                    f"batch {batch_num+1}/{total_batches}"
                )

                logging.info(
                    f"Checkpoint guardado | "
                    f"batch={batch_num+1}"
                )

            # Delay
            time.sleep(5)

    finally:

        # Final save
        print("\nGuardando resultados finales...")

        logging.info("Guardando resultados finales")
        
        results_df = pd.DataFrame(results)

        results_df = results_df.drop_duplicates(
            subset="id",
            keep="first"
        )

        results_df.to_csv(
            OUTPUT_PATH,
            index=False
        )

        results_df.to_json(
            OUTPUT_PATH_JSON,
            orient="records",
            force_ascii=False
        )

        # Merge with original dataset 
        df_final = df_original.merge(
            results_df,
            on="id",
            how="left"
        )

        df_final.to_csv(
            FINAL_OUTPUT_PATH,
            index=False
        )

        print("\n✅ Clasificación terminada")

        logging.info("Clasificación terminada")

        # Class counts
        counts = df_final["clasif"].value_counts()

        print("\nDistribución clases:")
        print(counts)

        logging.info(f"Distribución clases:\n{counts}")


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":

    main()

