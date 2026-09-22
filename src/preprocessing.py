## Dowloading et cleanning the dataset 
from mlcroissant import Dataset
import pandas as pd
import re
import unicodedata
import emoji
from nltk.corpus import stopwords
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
import pandas as pd


def preprocessing (code) :
    ## First, let's download our dataset

    ds = Dataset(jsonld="https://huggingface.co/api/datasets/shmuhammad/AfriSenti-twitter-sentiment/croissant")
    records = ds.records(code)

    #records_yoruba = ds.records("yor")

    # Convertir l'itérateur complet en DataFrame
    df = pd.DataFrame(records) # for pidding
    #df_yor = pd.DataFrame(records_yoruba) # for yoruba

    # Nettoyer le nom des colonnes pour enlever le préfixe "twi/"
    df.columns = [col.replace(code, "") for col in df.columns]
    #df_yor.columns = [col.replace("yor/", "") for col in df_yor.columns]


    # Ajouter une label lisibe 
    sentiment = {0 : "Neutre", 1 : "Positive", 2 : "Negative"}
    df['sentiment'] = df["label"].map(sentiment)
    #df_yor['sentiment'] = df_yor["label"].map(sentiment)

    df['split'] = df['split'].str.decode('utf-8')
    df['tweet'] = df['tweet'].str.decode('utf-8')


    return df


## fonction to normalyse data

def normalyse_data (text : str)-> str :

    # 1. Passer en minuscules
    text = text.lower()
    
     # 2. Convertir les emojis en texte brut (ex: "😊" -> ":smiling_face:")
    # language='en' permet d'avoir des descriptions universelles en anglais
    text = emoji.demojize(text, language='en')
    
    # 3. Transformer le format de la librairie (:mot_mot:) en votre tag (<emoji_mot_mot>)
    # On capture tout ce qui se trouve entre deux points ':' et on le met dans la balise
    text = re.sub(r':([a-zA-Z0-9_\-]+):', r'<emoji_\1>', text)
    
    # 4. Supprimer les URL et les @mentions Twitter
    text = re.sub(r"http\s+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text) # Retire le symbole # mais garde le mot
    
    # 5. Réduire les lettres répétées (ex: chaaaama -> chaama)
    text = re.sub(r'(.)\1+', r'\1\1', text)
    
    # 6. Normalisation Unicode (NFC)
    text = unicodedata.normalize('NFC', text)

    # Ne garder que les lettres et les espaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # 7. Nettoyer les espaces superflus
    text = re.sub(r'\s+', ' ', text).strip()

    whitelist = ["n't", "not", "no"]
    
    stopwords_list = stopwords.words('english')

    words = [word for word in text.split() if (word not in stopwords_list or word in whitelist) and len(word) > 2]
    
    return ' '.join(words)



def data_augmentation ( 
        df,
        configuration = None
                       ) :
    # 1. Définir précisément la taille voulue pour chaque couple (split, label)
    if configuration is None:
        configuration = {
            ("test", 1): 1500,
            ("test", 0): 900,
            ("train", 1): 3108,
            ("train", 0): 1200,
            ("validation", 1): 763,
            ("validation", 0): 400,
        }

    # 2. Filtrer et combiner selon la configuration
    df_yor_to_back = pd.concat(
        [
            df[(df["split"] == cat) & (df["label"] == lab)].head(taille)
            for (cat, lab), taille in configuration.items()
        ]
    ).reset_index(drop=True)

    return df_yor_to_back


def back_translate_dataset(df):
    # 1. Configuration du modèle et du tokenizer
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # 2. Préparation de votre dataset
    # (Remplacez cette partie par le chargement de votre fichier, ex: Dataset.from_pandas(pd.read_csv("data.csv")))

    dataset = Dataset.from_pandas(df)
    
    # 3. Codes de langue NLLB
    lang_src = "yor_Latn" # Français
    lang_tgt = "pcm_Latn" # Anglais (langue pivot)

    # 4. Fonction de traitement par lot (Batch Processing)
    def process_batch(batch):
        # --- ÉTAPE 1 : TRADUCTION ALLER (FR -> EN) ---
        tokenizer.src_lang = lang_src
        inputs_aller = tokenizer(batch["tweet"], padding=True, truncation=True, return_tensors="pt").to(device)
        
        tokens_aller = model.generate(
            **inputs_aller,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(lang_tgt),
            max_length=128
        )
        textes_intermediaires = tokenizer.batch_decode(tokens_aller, skip_special_tokens=True)
        
        # --- ÉTAPE 2 : TRADUCTION RETOUR (EN -> FR) ---
        tokenizer.src_lang = lang_tgt
        inputs_retour = tokenizer(textes_intermediaires, padding=True, truncation=True, return_tensors="pt").to(device)
        
        tokens_retour = model.generate(
            **inputs_retour,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(lang_src),
            max_length=128
        )
        textes_finals = tokenizer.batch_decode(tokens_retour, skip_special_tokens=True)
        
        # On retourne les nouvelles colonnes à ajouter au dataset
        return {
            "texte_intermediaire": textes_intermediaires,
            "back_translated": textes_finals
        }

    # 5. Application de la fonction sur tout le dataset
    # batched=True permet de traiter les phrases par groupes (ici 2 par 2 pour l'exemple)
    print("Début de la back-translation du dataset...")
    dataset_augmente = dataset.map(process_batch, batched=True, batch_size=2)
    
    # 6. Conversion et affichage/sauvegarde du résultat
    df_resultat = dataset_augmente.to_pandas()
    print("\n--- Résultat Final ---")
    #print(df_resultat[["texte_original", "back_translated"]])
    
    # Sauvegarde au format CSV
    df_resultat.to_csv("dataset_back_translated.csv", index=False)
    print("\nDataset sauvegardé sous 'dataset_back_translated.csv'")




def split_data(df: pd.DataFrame) :
    ## Extraction du train
    X_train = df[df['split'] == 'train']['tweet']
    y_train = df[df['split'] == 'train']['label']

    ##Extraction du test 
    X_test = df[df['split'] == 'test']['tweet']
    y_test = df[df['split'] == 'test']['label']

    ## Extraction du validation 
    X_val = df[df['split'] == 'validation']['tweet']
    y_val = df[df['split'] == 'validation']['label'] 

    return X_train, y_train, X_val, y_val, X_test, y_test