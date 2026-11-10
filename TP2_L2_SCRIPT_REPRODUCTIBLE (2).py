# ===== PHASE 01 =====
from pathlib import Path
import pandas as pd
import numpy as np
import html
import hashlib

colonnes_source = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country"
]
fichiers = sorted(Path("/kaggle/input").rglob("Online Retail.xlsx"))
assert len(fichiers) == 1, "Ajouter un seul fichier Online Retail.xlsx dans les données Kaggle."
source = fichiers[0]
sortie = Path("/kaggle/working")
sortie.mkdir(parents=True, exist_ok=True)
brut = pd.read_excel(source)
assert list(brut.columns) == colonnes_source, "Vérifier le contrat des huit colonnes."
brut["SourceIndex"] = np.arange(len(brut))
brut["SourceRow"] = brut["SourceIndex"] + 2
brut["SourceFile"] = source.name
empreinte = hashlib.sha256(source.read_bytes()).hexdigest()
print("Empreinte du fichier :", empreinte)
print("Lignes :", len(brut), "Colonnes métier :", len(colonnes_source))
print(brut[colonnes_source + ["SourceIndex", "SourceRow"]].head().to_string(index=False))

# ===== PHASE 02 =====
audit_initial = pd.DataFrame({
    "Colonne": colonnes_source,
    "Type_observe": [str(brut[c].dtype) for c in colonnes_source],
    "Valeurs_manquantes": [int(brut[c].isna().sum()) for c in colonnes_source]
})
print(audit_initial.to_string(index=False))
print("Commandes commençant par C :",
      brut["InvoiceNo"].astype(str).str.startswith("C").sum())
print("Copies exactes supplémentaires :",
      brut.duplicated(subset=colonnes_source, keep="first").sum())

# ===== PHASE 03 =====
masque_doublon = brut.duplicated(subset=colonnes_source, keep="first")
doublons_exacts = brut.loc[masque_doublon].copy()
staging_unique = brut.loc[~masque_doublon].copy()
assert len(brut) == len(doublons_exacts) + len(staging_unique)
assert not staging_unique.duplicated(subset=colonnes_source).any()
print("Copies retirées :", len(doublons_exacts))
print("Lignes uniques :", len(staging_unique))
print(doublons_exacts[["SourceIndex", "InvoiceNo", "StockCode", "Quantity"]].head().to_string(index=False))

# ===== PHASE 04 =====
staging_type = staging_unique.copy()
for c in ["StockCode", "Description", "Country"]:
    staging_type[c + "_norm"] = staging_type[c].astype("string").str.strip().str.upper()
staging_type["InvoiceNo_norm"] = staging_type["InvoiceNo"].astype("string").str.strip()
staging_type["CustomerID_norm"] = (
    staging_type["CustomerID"].astype("string").str.strip().replace("", pd.NA)
)
for c in ["Quantity", "UnitPrice"]:
    staging_type[c + "_num"] = pd.to_numeric(staging_type[c], errors="coerce")
    staging_type[c + "_echec"] = (
        staging_type[c].notna() & staging_type[c + "_num"].isna()
    )
# Le classeur fourni contient des dates Excel reconnues par read_excel.
if pd.api.types.is_datetime64_any_dtype(staging_type["InvoiceDate"]):
    staging_type["InvoiceDate_dt"] = pd.to_datetime(staging_type["InvoiceDate"], errors="coerce")
else:
    # Pour une source textuelle : convention explicite jour/mois/année.
    staging_type["InvoiceDate_dt"] = pd.to_datetime(
        staging_type["InvoiceDate"], format="%d/%m/%Y %H:%M", errors="coerce"
    )
staging_type["InvoiceDate_echec"] = (
    staging_type["InvoiceDate"].notna() & staging_type["InvoiceDate_dt"].isna()
)
anomalies_typage = staging_type.loc[
    staging_type[["Quantity_echec", "UnitPrice_echec", "InvoiceDate_echec"]].any(axis=1)
    | staging_type["Quantity_num"].isna()
    | staging_type["UnitPrice_num"].isna()
    | staging_type["InvoiceDate_dt"].isna()
].copy()
print(staging_type[["StockCode", "StockCode_norm", "Quantity", "Quantity_num"]].head().to_string(index=False))
print("Anomalies à diagnostiquer :", len(anomalies_typage))
assert len(staging_type) == len(staging_unique)
assert anomalies_typage.empty, "Ne pas inventer un filtre : montrer les anomalies à l'enseignante."

# ===== PHASE 05 =====
codes_non_commerciaux = {
    "POST": "FRAIS_PORT", "DOT": "FRAIS_PORT", "C2": "FRAIS_PORT",
    "M": "AJUSTEMENT", "B": "AJUSTEMENT", "PADS": "AJUSTEMENT",
    "D": "REMISE", "BANK CHARGES": "FRAIS_BANCAIRES",
    "AMAZONFEE": "COMMISSION", "CRUK": "COMMISSION",
    "S": "ECHANTILLON",
    "GIFT_0001_10": "BON_CADEAU", "GIFT_0001_20": "BON_CADEAU",
    "GIFT_0001_30": "BON_CADEAU", "GIFT_0001_40": "BON_CADEAU",
    "GIFT_0001_50": "BON_CADEAU"
}
integration = staging_type.copy()
code = integration["StockCode_norm"].fillna("")
description = integration["Description_norm"].fillna("")
integration["TypeLigne"] = code.map(codes_non_commerciaux).fillna("PRODUIT_COMMERCIAL")
integration["OrigineClassification"] = np.where(
    code.isin(codes_non_commerciaux), "REFERENTIEL_CODE", "REGLE_PAR_DEFAUT"
)
motif = (
    r"\b(?:ADJUSTMENT|MANUAL|POSTAGE|BANK CHARGES|AMAZON FEE|"
    r"DISCOUNT|SAMPLES?|COMMISSION|CARRIAGE|DOTCOM POSTAGE)\b"
)
suspect = description.str.contains(motif, regex=True, na=False)
a_verifier = suspect & integration["TypeLigne"].eq("PRODUIT_COMMERCIAL")
integration.loc[a_verifier, "TypeLigne"] = "A_VERIFIER"
integration.loc[a_verifier, "OrigineClassification"] = "DESCRIPTION_SUSPECTE"
classification_effectifs = integration["TypeLigne"].value_counts().rename_axis("TypeLigne").reset_index(name="Nombre")
assert len(integration) == len(staging_type)
assert classification_effectifs["Nombre"].sum() == len(integration)
print(classification_effectifs.to_string(index=False))
print(integration[["SourceIndex", "StockCode", "Description", "TypeLigne", "OrigineClassification"]].head().to_string(index=False))

# ===== PHASE 06 =====
integration["Annulation"] = integration["InvoiceNo_norm"].str.startswith("C", na=False)
flags = pd.DataFrame({
    "ANNULATION": integration["Annulation"],
    "QUANTITE_NON_POSITIVE": integration["Quantity_num"].le(0),
    "PRIX_NON_POSITIF": integration["UnitPrice_num"].le(0),
    "TYPE_HORS_VENTES": integration["TypeLigne"].ne("PRODUIT_COMMERCIAL")
}, index=integration.index)
# Pour chaque ligne, recueillir les noms des indicateurs vrais.
integration["MotifsExclusion"] = ""
for nom in flags.columns:
    positions = flags[nom]
    integration.loc[positions, "MotifsExclusion"] += nom + ";"
integration["MotifsExclusion"] = integration["MotifsExclusion"].str.rstrip(";")
integration["InclureFactVente"] = ~flags.any(axis=1)
print(integration[["SourceIndex", "TypeLigne", "InclureFactVente", "MotifsExclusion"]].head(12).to_string(index=False))
print("Effectifs par motif (chevauchements possibles) :")
print(flags.sum().to_string())

# ===== PHASE 07 =====
etape_0 = integration.copy()
etape_1 = etape_0.loc[~etape_0["Annulation"]].copy()
etape_2 = etape_1.loc[etape_1["Quantity_num"].gt(0)].copy()
etape_3 = etape_2.loc[etape_2["UnitPrice_num"].gt(0)].copy()
etape_4 = etape_3.loc[etape_3["TypeLigne"].eq("PRODUIT_COMMERCIAL")].copy()

def montant(df):
    return (df["Quantity_num"] * df["UnitPrice_num"]).sum()

etapes = [etape_0, etape_1, etape_2, etape_3, etape_4]
conditions = ["Staging dédoublonné", "InvoiceNo ne commence pas par C",
              "Quantity > 0", "UnitPrice > 0", "TypeLigne = PRODUIT_COMMERCIAL"]
journal = pd.DataFrame([
    {"Etape": i, "Condition_conservee": conditions[i],
     "Retirees": None if i == 0 else len(etapes[i-1]) - len(df),
     "Restantes": len(df), "Quantite": df["Quantity_num"].sum(),
     "Montant": montant(df),
     "Clients_absents": int(df["CustomerID_norm"].isna().sum())}
    for i, df in enumerate(etapes)
])
print(journal.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
preparee = etape_4.copy()
preparee["MontantLigne"] = preparee["Quantity_num"] * preparee["UnitPrice_num"]
audit_exclusions = integration.loc[~integration["InclureFactVente"]].copy()
assert set(preparee["SourceIndex"]) == set(integration.loc[integration["InclureFactVente"], "SourceIndex"])

# ===== PHASE 08 =====
attendues = integration.loc[
    (~integration["Annulation"])
    & (integration["Quantity_num"] > 0)
    & (integration["UnitPrice_num"] > 0)
    & integration["TypeLigne"].eq("PRODUIT_COMMERCIAL")
]
ids_attendus = set(attendues["SourceIndex"])
ids_observes = set(preparee["SourceIndex"])
ids_clients_absents = set(attendues.loc[attendues["CustomerID_norm"].isna(), "SourceIndex"])
assert preparee["SourceIndex"].is_unique
assert ids_observes == ids_attendus, "Ventes perdues ou ajoutées."
assert ids_clients_absents <= ids_observes, "Vente au client absent perdue."
assert set(preparee["SourceIndex"]).isdisjoint(set(audit_exclusions["SourceIndex"]))
assert set(integration["SourceIndex"]) == ids_observes | set(audit_exclusions["SourceIndex"])
assert len(brut) == len(doublons_exacts) + len(audit_exclusions) + len(preparee)
# Comparaison ligne par ligne ; une tolérance ne fixe pas une précision financière contractuelle.
controle_lignes = attendues[["SourceIndex", "Quantity_num", "UnitPrice_num"]].merge(
    preparee[["SourceIndex", "MontantLigne"]], on="SourceIndex", how="outer", validate="one_to_one", indicator=True
)
assert controle_lignes["_merge"].eq("both").all()
assert np.allclose(
    controle_lignes["Quantity_num"] * controle_lignes["UnitPrice_num"],
    controle_lignes["MontantLigne"], rtol=1e-12, atol=1e-9
)
assert np.isclose(montant(attendues), preparee["MontantLigne"].sum(), rtol=1e-12, atol=1e-9)
controle_final = pd.DataFrame({
    "Controle": ["Ventes préparées", "Lignes écartées après dédoublonnage",
                 "Quantité totale conservée", "Somme MontantLigne",
                 "Ventes conservées avec CustomerID manquant"],
    "Valeur_observee": [len(preparee), len(audit_exclusions),
                        preparee["Quantity_num"].sum(), preparee["MontantLigne"].sum(),
                        preparee["CustomerID_norm"].isna().sum()]
})
print(controle_final.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print("Provenances et montants : contrôles exécutés sans échec.")

# ===== PHASE 09 =====
preparee.to_csv(sortie / "TP2_L2_SOURCE_PREPAREE.csv", index=False, encoding="utf-8-sig")
integration.to_csv(sortie / "TP2_L2_STAGING_TRACE.csv", index=False, encoding="utf-8-sig")
audit_exclusions.to_csv(sortie / "TP2_L2_AUDIT_EXCLUSIONS.csv", index=False, encoding="utf-8-sig")
# Pièces de traçabilité complémentaires.
doublons_exacts.to_csv(sortie / "TP2_L2_DOUBLONS_EXACTS.csv", index=False, encoding="utf-8-sig")
journal.to_csv(sortie / "TP2_L2_JOURNAL_CONTROLES.csv", index=False, encoding="utf-8-sig")
rapport = """<!doctype html><html lang="fr"><meta charset="utf-8">
<title>M115 — Rapport qualité L2</title><style>body{font-family:Arial;max-width:1100px;margin:30px auto;line-height:1.5}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}textarea{width:95%;min-height:80px}@media print{textarea{border:1px solid #999}}</style>
<h1>M115 — TP2 · Rapport qualité L2</h1>
<p>Pr. AMINA ABOULMIRA · Étudiants : ______________________</p>"""
rapport += "<p>Fichier : " + html.escape(source.name) + "<br>SHA-256 : " + empreinte + "</p>"
for titre, tableau in [("Audit initial", audit_initial), ("Classification", classification_effectifs),
                       ("Journal des étapes", journal), ("Contrôle final", controle_final)]:
    rapport += "<h2>" + titre + "</h2>" + tableau.to_html(index=False, float_format=lambda x: f"{x:.2f}")
    rapport += '<p>Interprétation et anomalie éventuelle :</p><textarea></textarea>'
rapport += '<h2>Règles et preuves</h2><p>Joindre le script, commenter les copies exactes, le client absent et les motifs multiples. Le montant décrit le périmètre MVP, pas un CA comptable net.</p><p>Les graphiques sont facultatifs : les tableaux exacts restent prioritaires.</p><p>Pour conserver vos commentaires : imprimer ce rapport en PDF après saisie.</p></html>'
(sortie / "TP2_CORRECTION_IDEALE_M115.html").write_text(rapport, encoding="utf-8")
for nom in sorted(sortie.glob("TP2_*.csv")):
    print(nom.name)
print("TP2_CORRECTION_IDEALE_M115.html")
print("À joindre : TP2_L2_SCRIPT_REPRODUCTIBLE.py téléchargé depuis la fiche, et votre notebook exécuté.")