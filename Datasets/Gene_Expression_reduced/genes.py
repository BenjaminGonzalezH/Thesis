"""
validar_genes_gaf.py
--------------------
Valida si los nombres de genes del archivo de mapeo ENSG existen en el GAF
de Gene Ontology, específicamente en:
  - Columna 3  (índice 2): DB Object Symbol
  - Columna 10 (índice 9): DB Object Name (valores separados por '|')

Uso:
    python3 validar_genes_gaf.py --csv archivo.csv --gaf archivo.gaf[.gz] --out resultados.csv

Argumentos opcionales:
    --csv-ensg   Nombre de la columna ENSG en el CSV  (default: 'incoming')
    --csv-name   Nombre de la columna gen en el CSV   (default: 'name')
"""

import argparse
import gzip
import csv
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv",      required=True,  help="Archivo CSV con el mapeo ENSG → nombre")
    parser.add_argument("--gaf",      required=True,  help="Archivo GAF (puede ser .gz)")
    parser.add_argument("--out",      default="resultados_validacion.csv", help="Archivo de salida CSV")
    parser.add_argument("--csv-ensg", default="incoming", help="Columna ENSG en el CSV (default: incoming)")
    parser.add_argument("--csv-name", default="name",     help="Columna nombre en el CSV (default: name)")
    return parser.parse_args()


def open_gaf(path: str):
    """Abre el GAF, con soporte para .gz."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def build_gaf_sets(gaf_path: str):
    """
    Lee el GAF y construye dos sets:
      - symbols : todos los valores únicos de la columna 3 (DB Object Symbol)
      - names   : todos los tokens únicos de la columna 10 (DB Object Name, sep='|')
    """
    symbols = set()
    names   = set()

    with open_gaf(gaf_path) as fh:
        for line in fh:
            if line.startswith("!"):   # líneas de cabecera/comentario
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 10:
                continue

            symbol = cols[2].strip()
            if symbol:
                symbols.add(symbol)

            # col 10 puede tener varios nombres separados por '|'
            for token in cols[9].split("|"):
                token = token.strip()
                if token:
                    names.add(token)

    return symbols, names


def classify(gene_name: str, symbols: set, names: set) -> str:
    """
    Devuelve una etiqueta de clasificación para el gen:
      in_symbol_and_name  → aparece en ambas columnas
      in_symbol_only      → sólo en DB Object Symbol (col 3)
      in_name_only        → sólo en DB Object Name   (col 10)
      not_found           → no aparece en ninguna
    """
    in_sym  = gene_name in symbols
    in_name = gene_name in names

    if in_sym and in_name:
        return "in_symbol_and_name"
    elif in_sym:
        return "in_symbol_only"
    elif in_name:
        return "in_name_only"
    else:
        return "not_found"


def main():
    args = parse_args()

    # ── 1. Cargar el CSV de mapeo ──────────────────────────────────────────────
    print(f"[1/3] Leyendo CSV: {args.csv}", flush=True)
    genes = []
    with open(args.csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ensg = row[args.csv_ensg].strip()
            name = row[args.csv_name].strip()
            if ensg and name:
                genes.append((ensg, name))
    print(f"      {len(genes):,} genes cargados.", flush=True)

    # ── 2. Construir los sets desde el GAF ────────────────────────────────────
    print(f"[2/3] Indexando GAF: {args.gaf}", flush=True)
    symbols, names = build_gaf_sets(args.gaf)
    print(f"      {len(symbols):,} símbolos únicos (col 3)", flush=True)
    print(f"      {len(names):,}   nombres únicos  (col 10)", flush=True)

    # ── 3. Clasificar y escribir resultados ───────────────────────────────────
    print(f"[3/3] Clasificando genes y escribiendo: {args.out}", flush=True)
    counts = {"in_symbol_and_name": 0, "in_symbol_only": 0,
              "in_name_only": 0, "not_found": 0}

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ensg_id", "gene_name", "status"])
        for ensg, name in genes:
            status = classify(name, symbols, names)
            counts[status] += 1
            writer.writerow([ensg, name, status])

    # ── Resumen ───────────────────────────────────────────────────────────────
    total = len(genes)
    print("\n── Resumen ────────────────────────────────────────")
    for label, n in counts.items():
        pct = 100 * n / total if total else 0
        print(f"  {label:<22} {n:>8,}  ({pct:.1f}%)")
    print(f"  {'TOTAL':<22} {total:>8,}")
    print(f"\nResultados guardados en: {args.out}")


if __name__ == "__main__":
    main()