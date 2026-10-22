# -*- coding: utf-8 -*-
"""
Simulation des risques - Fraudes aux examens (Activité 4, étapes 10 à 16)
Bibliothèques : pandas, seaborn, matplotlib (Google Colab ready)

Utilisation sous Colab :
    1. Importer le fichier Excel (ou le lien CSV Google Sheets des réponses)
    2. Modifier FICHIER ci-dessous puis exécuter toutes les cellules.
Tout est paramétré dans la section PARAMÈTRES : changer les données d'entrée
met automatiquement à jour tous les résultats et graphiques.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============================ PARAMÈTRES ============================
FICHIER = "Simulation_Risques.xlsx"          # ou URL .csv d'un Google Sheets publié
FEUILLE = "Réponses au formulaire 1"
Z = 1.96                                      # IC 95 %
SEUILS_IMPACT = [0.0, 0.2, 0.4, 0.6, 0.8]     # indices 1..5
SEUILS_PROBA = [0.0, 0.25, 0.5, 0.75]         # indices 1..4
IND_MAX_IMPACT, IND_MAX_PROBA = 5, 4

# Colonnes du formulaire (ordre Google Forms)
COL = {"Q5": 5, "Q6": 6, "Q7": 7, "Q8": 8, "Q9": 9, "Q10": 10, "Q11": 11, "Q12": 12,
       "Q13": 13, "Q14": 14, "Q15": 15, "Q16": 16, "Q17": 17, "Q18": 18, "Q19": 19}

# Questions à choix multiples : code -> libellé recherché dans la réponse
MULTI = {
    "Q5": ["Peur d'échouer", "Recherche d'une bonne note", "Pression familiale",
           "Importance de l'évaluation", "Charge de travail"],
    "Q8": ["Procrastination", "Charge de travail", "Mauvaise organisation",
           "Nombre élevé de matières", "Travail personnel/professionnel"],
    "Q11": ["Encouragement direct", "Partage de réponses", "Normalisation de la fraude",
            "Pression du groupe", "Témoignage de fraudes précédentes"],
    "Q13": ["Règles peu expliquées", "Manque de sensibilisation", "Règles complexes",
            "Différences entre enseignants", "Manque d'information"],
    "Q15": ["Difficulté du contenu", "Temps insuffisant", "Questions complexes",
            "Volume important du programme", "Manque de préparation"],
    "Q17": ["Surveillance limitée", "Grand nombre d'étudiants", "Absence de contrôle technologique",
            "Examens à distance", "Manque de sanctions visibles"],
    "Q19": ["Pression pour obtenir de bonnes notes", "Manque de temps et mauvaise organisation",
            "Influence des autres étudiants", "Manque de compréhension des règles",
            "Difficulté des examens et des cours", "Faible risque d'être détecté"],
}
# Échelles (min, max, inversion)
ECHELLE = {"Q6": (1, 5, False), "Q7": (0, 4, False), "Q9": (1, 5, False), "Q10": (1, 5, False),
           "Q12": (1, 5, True), "Q14": (1, 5, False), "Q16": (1, 5, True), "Q18": (1, 5, False)}
Q6_MAP = [("Très fortement", 5), ("Fortement", 4), ("Moyennement", 3), ("Faiblement", 2), ("Pas du tout", 1)]
Q7_MAP = [("trés important", 4), ("très important", 4), ("important", 3), ("modéré", 2),
          ("faible", 1), ("Aucun", 0)]

# Causes racines (6M) : (ID, famille, cause, [questions impact], [questions probabilité])
CAUSES = [
    ("C1", "Main-d'œuvre", "Pression pour bonnes notes", ["Q6", "Q19/1"], ["Q5/1", "Q5/2", "Q5/3", "Q5/4"]),
    ("C2", "Main-d'œuvre", "Manque de temps", ["Q7", "Q19/2"], ["Q8/5", "Q15/2"]),
    ("C3", "Main-d'œuvre", "Mauvaise organisation", ["Q9", "Q19/2"], ["Q8/1", "Q8/3"]),
    ("C4", "Méthode", "Difficulté des examens", ["Q14", "Q19/5"], ["Q15/3", "Q15/5"]),
    ("C5", "Méthode", "Charge de travail élevée", ["Q19/5"], ["Q5/5", "Q8/2", "Q8/4"]),
    ("C6", "Matériel", "Difficulté des cours", ["Q19/5"], ["Q15/1", "Q15/4"]),
    ("C7", "Milieu", "Influence des autres étudiants", ["Q10", "Q19/3"], ["Q11/2", "Q11/3", "Q11/4", "Q11/5"]),
    ("C8", "Milieu", "Encouragement des amis", ["Q10", "Q19/3"], ["Q11/1"]),
    ("C9", "Management", "Manque de compréhension des règles", ["Q19/4"],
     ["Q12", "Q13/1", "Q13/2", "Q13/3", "Q13/4", "Q13/5"]),
    ("C10", "Management", "Faible risque d'être détecté", ["Q19/6"], ["Q16", "Q17/1", "Q17/2"]),
    ("C11", "Management", "Manque de sanctions visibles", ["Q18"], ["Q17/5"]),
    ("C12", "Machine", "Absence de contrôle technologique", ["Q19/6"], ["Q17/3"]),
    ("C13", "Machine", "Examens à distance", ["Q19/6"], ["Q17/4"]),
]


# ============================ CODAGE ============================
def lire_reponses(fichier=FICHIER, feuille=FEUILLE):
    if str(fichier).lower().endswith(".csv") or str(fichier).startswith("http"):
        df = pd.read_csv(fichier)
    else:
        df = pd.read_excel(fichier, sheet_name=feuille)
    return df.dropna(how="all")


def coder(df):
    """Transforme les réponses brutes en variables numériques (équivalent de la feuille Codage)."""
    cols = df.columns
    out = pd.DataFrame(index=df.index)
    for q, libs in MULTI.items():
        serie = df[cols[COL[q]]].astype(str)
        for i, lib in enumerate(libs, 1):
            out[f"{q}/{i}"] = serie.str.contains(lib, case=False, regex=False).astype(int)
    txt6 = df[cols[COL["Q6"]]].astype(str)
    out["Q6"] = txt6.map(lambda t: next((v for k, v in Q6_MAP if k.lower() in t.lower()), np.nan))
    txt7 = df[cols[COL["Q7"]]].astype(str)
    out["Q7"] = txt7.map(lambda t: next((v for k, v in Q7_MAP if k.lower() in t.lower()), np.nan))
    for q in ["Q9", "Q10", "Q12", "Q14", "Q16", "Q18"]:
        out[q] = pd.to_numeric(df[cols[COL[q]]], errors="coerce")
    return out


# ============================ ÉTAPE 10 : STATISTIQUES ============================
def statistiques(cod):
    n = cod.count()
    m, s = cod.mean(), cod.std(ddof=1)
    st = pd.DataFrame({"n": n, "moyenne": m, "ecart_type": s})
    st["ic_inf"] = m - Z * s / np.sqrt(n)
    st["ic_sup"] = m + Z * s / np.sqrt(n)
    binaire = [q for q in st.index if q not in ECHELLE]   # proportions : bornées à [0, 1]
    st.loc[binaire, "ic_inf"] = st.loc[binaire, "ic_inf"].clip(lower=0)
    st.loc[binaire, "ic_sup"] = st.loc[binaire, "ic_sup"].clip(upper=1)
    return st


def pourcentage(q, moyenne):
    if q in ECHELLE:
        lo, hi, inv = ECHELLE[q]
        p = (moyenne - lo) / (hi - lo)
        return 1 - p if inv else p
    return moyenne  # variables binaires : proportion de répondants


def indice(valeur, seuils):
    return int(np.searchsorted(seuils, valeur, side="right"))  # [0,s1[ -> 1, ...


def calculer(cod):
    st = statistiques(cod)
    pct = pd.Series({q: pourcentage(q, st.loc[q, "moyenne"]) for q in st.index})
    lignes = []
    for cid, fam, cause, qi, qp in CAUSES:
        y = pct[qi].mean()
        x = pct[qp].mean()
        iy, ix = indice(y, SEUILS_IMPACT), indice(x, SEUILS_PROBA)
        lignes.append(dict(ID=cid, Famille=fam, Cause=cause, Yi_pct=y, Xi_pct=x,
                           Indice_Impact=iy, Indice_Proba=ix))
    r = pd.DataFrame(lignes)
    # Étape 12 : zones (milieu des intervalles d'indices)
    mx = (1 + IND_MAX_PROBA) / 2
    my = (1 + IND_MAX_IMPACT) / 2
    def zone(x, y):
        if x > mx and y > my: return "I"
        if x <= mx and y > my: return "II"
        if x <= mx and y <= my: return "III"
        return "IV"
    r["Zone"] = [zone(x, y) for x, y in zip(r.Indice_Proba, r.Indice_Impact)]
    # Étape 14 : coûts
    r["Cout_pct"] = r.Xi_pct * r.Yi_pct
    r["Cout_indices"] = r.Indice_Proba * r.Indice_Impact
    r["Rang"] = r.Cout_pct.rank(ascending=False, method="min").astype(int)
    return st, r, (mx, my)


# ============================ GRAPHIQUES ============================
def carte(r, mid, fichier="cartographie_risques.png"):
    mx, my = mid
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.axvspan(mx, IND_MAX_PROBA + .5, ymin=(my - .5) / (IND_MAX_IMPACT), ymax=1, color="#f4c7c3", alpha=.5)
    ax.axvline(mx, color="grey", ls="--"); ax.axhline(my, color="grey", ls="--")
    rng = np.random.default_rng(3)
    jit = rng.uniform(-.12, .12, (len(r), 2))
    couleurs = dict(zip(sorted(r.Famille.unique()), sns.color_palette("tab10", r.Famille.nunique())))
    for (i, row), j in zip(r.iterrows(), jit):
        ax.scatter(row.Indice_Proba + j[0], row.Indice_Impact + j[1], s=140,
                   color=couleurs[row.Famille], edgecolor="k", zorder=3)
        ax.annotate(row.ID, (row.Indice_Proba + j[0] + .05, row.Indice_Impact + j[1] + .05), fontsize=9)
    for lab, (x, y) in {"I": (3.55, 5.3), "II": (.7, 5.3), "III": (.7, .6), "IV": (3.55, .6)}.items():
        ax.text(x, y, f"Zone {lab}", fontsize=12, weight="bold", color="grey")
    ax.set_xlim(.5, IND_MAX_PROBA + .5); ax.set_ylim(.5, IND_MAX_IMPACT + .5)
    ax.set_xticks(range(1, IND_MAX_PROBA + 1)); ax.set_yticks(range(1, IND_MAX_IMPACT + 1))
    ax.set_xlabel("Probabilité d'occurrence (indice Xi)"); ax.set_ylabel("Impact (indice Yi)")
    ax.set_title("Cartographie des risques (léger décalage visuel des points superposés)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markeredgecolor="k", label=f) for f, c in couleurs.items()]
    ax.legend(handles=handles, title="6M", loc="center left", bbox_to_anchor=(1, .5))
    fig.tight_layout(); fig.savefig(fichier, dpi=200); plt.close(fig)


def carte_pct(r, fichier="cartographie_pct.png"):
    """Version lisible : coordonnées en pourcentages (points non superposés)."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    sns.scatterplot(data=r, x="Xi_pct", y="Yi_pct", hue="Famille", s=160, edgecolor="k", ax=ax, zorder=3)
    for _, row in r.iterrows():
        ax.annotate(row.ID, (row.Xi_pct, row.Yi_pct), xytext=(6, 5), textcoords="offset points", fontsize=9)
    ax.axvline(SEUILS_PROBA[2], color="grey", ls="--"); ax.axhline(SEUILS_IMPACT[3], color="grey", ls="--")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Probabilité d'occurrence Xi (%)"); ax.set_ylabel("Impact Yi (%)")
    ax.set_title("Cartographie des risques - valeurs en pourcentage")
    ax.legend(title="6M", loc="center left", bbox_to_anchor=(1, .5))
    fig.tight_layout(); fig.savefig(fichier, dpi=200); plt.close(fig)


def barres_cout(r, fichier="cout_risques.png"):
    d = r.sort_values("Cout_pct")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.barplot(data=d, y="Cause", x="Cout_pct", hue="Famille", dodge=False, ax=ax)
    ax.set_xlabel("Coût du risque Ci = Xi × Yi"); ax.set_ylabel("")
    ax.set_title("Classement des risques par coût")
    fig.tight_layout(); fig.savefig(fichier, dpi=200); plt.close(fig)


def ishikawa(r, fichier="ishikawa_enrichi.png"):
    fams = ["Main-d'œuvre", "Méthode", "Matériel", "Milieu", "Management", "Machine"]
    fig, ax = plt.subplots(figsize=(15, 8)); ax.axis("off"); ax.set_xlim(0, 15); ax.set_ylim(0, 8)
    ax.annotate("", xy=(13, 4), xytext=(.5, 4), arrowprops=dict(arrowstyle="-|>", lw=3, color="#333"))
    ax.text(13.1, 4, "Fraudes aux\nexamens", va="center", fontsize=12, weight="bold",
            bbox=dict(boxstyle="round", fc="#f4c7c3"))
    xs = [2.5, 6.5, 10.5]
    for k, f in enumerate(fams):
        top = k < 3
        x0 = xs[k % 3]
        y0 = 7.2 if top else .8
        ax.plot([x0 - 1.2, x0 + .8], [y0, 4], color="#555", lw=2)
        ax.text(x0 - 1.2, y0 + (.35 if top else -.35), f, ha="center", fontsize=12, weight="bold",
                color="white", bbox=dict(boxstyle="round", fc="#1f4e79"))
        sub = r[r.Famille == f]
        for n, (_, row) in enumerate(sub.iterrows()):
            yy = y0 + (-0.8 - n * 0.95) if top else y0 + 0.8 + n * 0.95
            xx = x0 - 1.2 + (x0 + .8 - (x0 - 1.2)) * (y0 - yy) / (y0 - 4)
            col = "#c0392b" if row.Zone in ("I", "II") else "#2c3e50"
            ax.text(xx - .1, yy, f"{row.ID} {row.Cause}\nXi={row.Xi_pct:.0%} · Yi={row.Yi_pct:.0%}",
                    ha="right", va="center", fontsize=8, color=col,
                    bbox=dict(boxstyle="round,pad=.2", fc="white", ec="#bbb"))
    ax.set_title("Diagramme d'Ishikawa enrichi (Xi = probabilité, Yi = impact ; rouge = zones I/II)", fontsize=13)
    fig.tight_layout(); fig.savefig(fichier, dpi=200); plt.close(fig)


# ============================ PRINCIPAL ============================
def main(fichier=FICHIER):
    df = lire_reponses(fichier)
    cod = coder(df)
    st, r, mid = calculer(cod)
    print(f"{len(df)} réponses analysées\n")
    print(st.round(3).to_string())
    print()
    print(r.round(3).to_string(index=False))
    # Étape 13 : zone la plus prioritaire non vide (ordre I > II > IV > III)
    for z in ["I", "II", "IV", "III"]:
        if (r.Zone == z).any():
            print(f"\nZone prioritaire non vide : {z} -> {', '.join(r[r.Zone == z].Cause)}")
            break
    top = r.loc[r.Cout_pct.idxmax()]
    print(f"Risque prioritaire (coût max) : {top.ID} {top.Cause} (Ci = {top.Cout_pct:.3f})")
    carte(r, mid); carte_pct(r); barres_cout(r); ishikawa(r)
    st.to_csv("statistiques.csv"); r.to_csv("resultats_risques.csv", index=False)
    return st, r


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else FICHIER)
