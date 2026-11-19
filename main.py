import ollama
import json
import csv
import re
from sklearn.metrics import f1_score


JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["sports", "economy", "technology",
                     "politics", "arts", "other"]
        },
        "entities": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5
        }
    },
    "required": ["category", "entities"],
    "additionalProperties": False
}


toplam = 0
basarili = 0
hatali = 0
temizleme_basarili = 0
dogru_tahmin = 0
yanlis_tahmin = 0
schema_basarili = 0
schema_hatali = 0
gercek_kategoriler = []
tahmin_kategoriler = []

# CSV dosyasını açıyoruz.

with open("dataset.csv", "r", encoding="utf-8-sig") as dosya:
    okuyucu = csv.DictReader(dosya)

    # Her satırı sırayla işliyoruz.

    for satir in okuyucu:
        toplam += 1
        metin = satir["text"]
        gercek_kategori = satir["expected_category"]
        print(f"\n========== METİN {toplam}/100 ==========", flush=True)

        print("\n--------------------------")
        print("Metin:", metin)
        print("Gerçek kategori:", gercek_kategori)

        # Metni Ollama'ya gönderiyoruz.

        cevap = ollama.chat(
            model="llama3.2:3b",
           # format=JSON_SCHEMA,
            messages=[
                {
                    "role": "user",
                    "content": f"""
Analyze the following text:

{metin}

Return ONLY valid JSON with these fields:

- category: Choose exactly one of:
  sports, economy, technology, politics, arts, other


- entities: Return at most 5 important proper names
  mentioned in the text, such as people,
  organizations, locations, and products.
  If none are found, return an empty list.

  
Do not write explanations.
"""
                }
            ],
        options={"temperature": 1.5, "seed": 42} )
        sonuc = cevap["message"]["content"]

       
        try:
                veri = json.loads(sonuc)
                basarili += 1
                print("JSON ilk denemede başarılı!")

        except json.JSONDecodeError:
                hatali += 1
                print("JSON hatalı! Temizleme deneniyor...")

                temiz_sonuc = re.sub(
                    r"^```(?:json)?\s*|\s*```$",
                    "",
                    sonuc.strip(),
                    flags=re.IGNORECASE
                ).strip()
                if temiz_sonuc.endswith("}}"):
                     temiz_sonuc = temiz_sonuc[:-1]

                
                try:
                    veri = json.loads(temiz_sonuc)
                    temizleme_basarili += 1
                    print("JSON temizlendikten sonra başarılı!")

                except json.JSONDecodeError:
                    veri = None
                    print("JSON temizlendikten sonra da hatalı!")


        if isinstance(veri, dict):
                print(json.dumps(veri, indent=4, ensure_ascii=False))

                tahmin_kategori = veri.get("category", "")
                gercek_kategoriler.append(gercek_kategori)
                tahmin_kategoriler.append(tahmin_kategori)

                print("Tahmin edilen kategori:", tahmin_kategori)

                if tahmin_kategori == gercek_kategori:
                 dogru_tahmin += 1
                 print("Kategori tahmini: DOĞRU")
                else:
                 yanlis_tahmin += 1
                 print("Kategori tahmini: YANLIŞ")


                # JSON Schema Validity kontrolü
                kategoriler = ["sports", "economy", "technology",
                "politics", "arts", "other"]

                schema_dogru = (
                isinstance(veri, dict)
                and set(veri.keys()) == {"category", "entities"}
                and isinstance(veri.get("category"), str)
                and veri["category"] in kategoriler
                and isinstance(veri.get("entities"), list)
                and len(veri["entities"]) <= 5
                and all(isinstance(e, str) for e in veri["entities"])
    )

                if schema_dogru:
                  schema_basarili += 1
                  print("Schema Validity: BASARILI")
                else:
                  schema_hatali += 1
                  print("Schema Validity: HATALI")



print("\n========== DEĞERLENDİRME ==========")

print("Toplam metin:", toplam)
print("İlk denemede başarılı:", basarili)
print("İlk denemede hatalı:", hatali)

if toplam > 0:
    basari_orani = (basarili / toplam) * 100
    print(f"İlk deneme başarı oranı: %{basari_orani:.2f}")


print("Temizleme ile kurtarılan:", temizleme_basarili)

toplam_basarili = basarili + temizleme_basarili
kalan_hatali = toplam - toplam_basarili

print("Temizleme sonrası toplam başarılı:", toplam_basarili)
print("Temizleme sonrası kalan hatalı:", kalan_hatali)

if toplam > 0:
    temizleme_orani = (toplam_basarili / toplam) * 100
    print(f"Temizleme sonrası başarı oranı: %{temizleme_orani:.2f}")



print("\n========== KATEGORİ DEĞERLENDİRMESİ ==========")

print("Doğru kategori tahmini:", dogru_tahmin)
print("Yanlış kategori tahmini:", yanlis_tahmin)

degerlendirilen = dogru_tahmin + yanlis_tahmin

if degerlendirilen > 0:
    kategori_basari = (dogru_tahmin / degerlendirilen) * 100
    print(f"Kategori tahmin başarı oranı: %{kategori_basari:.2f}")



print("\n========== SCHEMA VALIDITY ==========")

print("Schema başarılı:", schema_basarili)
print("Schema hatalı:", schema_hatali)

schema_toplam_hatali = schema_hatali + (toplam - toplam_basarili)

print("JSON okunamayan:", toplam - toplam_basarili)

if toplam > 0:
    schema_orani = (schema_basarili / toplam) * 100
    print(f"Schema başarı oranı: %{schema_orani:.2f}")



print("\n========== CLASSIFICATION F1 ==========")

if gercek_kategoriler:
    f1 = f1_score(
        gercek_kategoriler,
        tahmin_kategoriler,
        labels=["sports", "economy", "technology",
                "politics", "arts", "other"],
        average="macro",
        zero_division=0
    )

    print(f"Classification Macro F1: {f1:.4f}")


print("===================================")